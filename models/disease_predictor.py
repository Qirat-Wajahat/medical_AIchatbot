"""Scenario follow-up retrieval helper.

This project intentionally separates:
- `data/scenarios.txt`: communication style only (follow-up questions)
- `data/medicines.json`: medical knowledge base (handled in app_flask.py)

`DiseasePredictionModel` remains for backward compatibility, but it no longer
trains or persists any ML model.
"""

import json
import re


class DiseasePredictionModel:
    """Scenario follow-up retriever (style-only).

    Note: disease prediction is handled via `data/medicines.json` in `app_flask.py`.
    """

    def __init__(
        self,
        data_paths=None,
        data_path=None,
        scenario_path=None,
        scenario_paths=None,
        scenario_label_aliases=None,
        use_scenarios_for_training: bool = False,
    ):
        """
        Initialize the disease prediction model

        Args:
            data_paths (str or list): Path or list of paths to medicines/diseases JSON files
            data_path (str): Backwards-compatible alias for a single path
            scenario_path (str): Backwards-compatible alias for a single scenarios text path
            scenario_paths (str or list): Path(s) to scenarios text files to use as extra training samples
            scenario_label_aliases (dict): Optional mapping from scenario labels -> canonical disease names
        """
        if data_paths is None and data_path is not None:
            data_paths = data_path
        if data_paths is None:
            data_paths = ['data/medicines.json']
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if scenario_paths is None and scenario_path is not None:
            scenario_paths = scenario_path
        if isinstance(scenario_paths, str):
            scenario_paths = [scenario_paths]

        self.data_paths = data_paths
        self.scenario_paths = scenario_paths or []
        # Kept for backwards compatibility, but intentionally ignored.
        self.use_scenarios_for_training = bool(use_scenarios_for_training)
        self._scenario_label_aliases = scenario_label_aliases or {}

        # Optional legacy fields (not used by the Flask app flow).
        self.diseases = []
        self.symptoms_map = {}

        # Scenario dialog samples used for style follow-ups.
        # Each entry: {"patient_text": str, "tokens": set[str], "doctor_lines": list[str]}
        self._scenario_dialog_samples = []

        # Load legacy disease data (best-effort). This project uses medicines.json differently,
        # so this usually results in an empty dataset and that's OK.
        self.load_data()

        # Load dialog samples from scenarios (style-only).
        for sp in self.scenario_paths:
            self._scenario_dialog_samples.extend(self._load_scenarios(sp))

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lightweight normalization shared by training and inference.

        Keeps spaces, removes punctuation/numbers, lowercases.
        """
        t = (text or "").lower()
        t = re.sub(r"http\S+|www\S+", " ", t)
        t = re.sub(r"[^a-z\s]", " ", t)
        t = " ".join(t.split())
        return t

    def _load_scenarios(self, scenario_path: str):
        """Parse a scenarios.txt file and return dialog snippets.

        We only use scenarios as a communication-style source of follow-up questions.
        """
        dialog_samples = []
        if not scenario_path:
            return dialog_samples

        try:
            with open(scenario_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return dialog_samples

        blocks = re.split(r"\n\s*Scenario\s+\d+\s*\n", "\n" + content.strip(), flags=re.IGNORECASE)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            patient_lines = []
            doctor_lines = []
            for line in block.splitlines():
                line = line.strip()
                if line.lower().startswith('patient:'):
                    patient_lines.append(line.split(':', 1)[1].strip())
                elif line.lower().startswith('doctor:'):
                    doctor_lines.append(line.split(':', 1)[1].strip())

            patient_text = self.normalize_text(' '.join(patient_lines))
            if not patient_text or not doctor_lines:
                continue

            tokens = set(patient_text.split())
            if not tokens:
                continue

            dialog_samples.append({
                'patient_text': patient_text,
                'tokens': tokens,
                'doctor_lines': doctor_lines,
            })

        return dialog_samples
    
    def load_data(self):
        """Load disease and symptom data from one or more JSON files and merge duplicates."""
        merged = {}

        for path in self.data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data.get('diseases', []) if isinstance(data, dict) else data

                    for disease in items:
                        if not isinstance(disease, dict):
                            continue

                        # Normalize name
                        name = disease.get('name', '').strip()
                        if not name:
                            continue

                        # Guard: only accept objects that actually look like a disease entry.
                        symptoms = disease.get('symptoms')
                        if not isinstance(symptoms, list) or len(symptoms) == 0:
                            continue

                        if name in merged:
                            # Merge symptoms
                            existing = merged[name]
                            existing_symptoms = set(existing.get('symptoms', []))
                            for s in disease.get('symptoms', []):
                                if s not in existing_symptoms:
                                    existing.setdefault('symptoms', []).append(s)
                                    existing_symptoms.add(s)

                            # Merge medicines by name
                            existing_meds = existing.setdefault('medicines', [])
                            existing_med_names = set(m.get('name') for m in existing_meds)
                            for m in disease.get('medicines', []):
                                if m.get('name') not in existing_med_names:
                                    existing_meds.append(m)
                                    existing_med_names.add(m.get('name'))
                        else:
                            # Add a shallow copy to avoid modifying source
                            merged[name] = {
                                'name': name,
                                'symptoms': list(symptoms),
                                'medicines': list(disease.get('medicines', []))
                            }
            except FileNotFoundError:
                print(f"Warning: Could not find data file at {path}")
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON file at {path}")

        # Convert merged dict to list
        self.diseases = list(merged.values())

        # Build symptom -> diseases mapping
        self.symptoms_map = {}
        for disease in self.diseases:
            disease_name = disease['name']
            for symptom in disease.get('symptoms', []):
                if symptom not in self.symptoms_map:
                    self.symptoms_map[symptom] = []
                self.symptoms_map[symptom].append(disease_name)

    def get_scenario_followups(self, text: str, top_k: int = 2):
        """Return doctor follow-up lines from the most similar scenarios.

        Similarity is computed via token overlap against the scenario's patient text.
        """
        if not self._scenario_dialog_samples or top_k <= 0:
            return []

        user_tokens = set(self.normalize_text(text).split())
        if not user_tokens:
            return []

        scored = []
        for idx, sample in enumerate(self._scenario_dialog_samples):
            tokens = sample.get('tokens') or set()
            if not tokens:
                continue
            common = len(user_tokens & tokens)
            if common <= 0:
                continue
            # Prefer higher overlap, then slightly prefer smaller scenarios (more specific).
            overlap = common / max(1, len(tokens))
            scored.append((overlap, common, -len(tokens), -idx))

        if not scored:
            return []

        scored.sort(reverse=True)
        followups = []
        for _, __, ___, neg_idx in scored:
            sample = self._scenario_dialog_samples[-neg_idx]
            for line in (sample.get('doctor_lines') or []):
                if line:
                    followups.append(line)
                    if len(followups) >= top_k:
                        return followups[:top_k]

        return followups[:top_k]