"""
Medical Reference Utilities
Loads and retrieves medical information from reference text
"""

import os


class MedicalReference:
    """Handles loading and retrieval of medical reference information"""
    
    def __init__(self, reference_path='data/medical_reference.txt'):
        """
        Initialize medical reference
        
        Args:
            reference_path (str): Path to medical reference text file
        """
        self.reference_path = reference_path
        self.reference_data = {}
        self.full_text = ""
        self.load_reference()
    
    def load_reference(self):
        """Load medical reference information from file"""
        try:
            with open(self.reference_path, 'r') as f:
                self.full_text = f.read()
                
            # Parse the reference text into sections
            self.parse_reference()
            
        except FileNotFoundError:
            print(f"Warning: Reference file not found at {self.reference_path}")
            self.full_text = ""
    
    def parse_reference(self):
        """Parse reference text into disease sections"""
        sections = self.full_text.split('\n\n')
        current_disease = None
        current_content = []
        
        for section in sections:
            lines = section.strip().split('\n')
            if len(lines) > 0:
                # Check if this is a disease header (all caps or with dashes)
                first_line = lines[0].strip()
                if first_line and (first_line.isupper() or '-' * 3 in section):
                    # Save previous disease if exists
                    if current_disease and current_content:
                        self.reference_data[current_disease] = '\n'.join(current_content)
                    
                    # Start new disease
                    current_disease = first_line.replace('-', '').strip()
                    current_content = lines[1:] if len(lines) > 1 else []
                elif current_disease:
                    current_content.extend(lines)
        
        # Save last disease
        if current_disease and current_content:
            self.reference_data[current_disease] = '\n'.join(current_content)
    
    def get_disease_info(self, disease_name):
        """
        Get reference information for a specific disease
        
        Args:
            disease_name (str): Name of the disease
            
        Returns:
            str: Reference information or None if not found
        """
        # Try exact match
        for key in self.reference_data.keys():
            if disease_name.lower() in key.lower() or key.lower() in disease_name.lower():
                return self.reference_data[key]
        
        return None
    
    def get_all_info(self):
        """
        Get all reference information
        
        Returns:
            str: Full reference text
        """
        return self.full_text
    
    def search_reference(self, keyword):
        """
        Search reference text for keyword
        
        Args:
            keyword (str): Search keyword
            
        Returns:
            list: List of sections containing the keyword
        """
        results = []
        keyword_lower = keyword.lower()
        
        for disease, content in self.reference_data.items():
            if keyword_lower in disease.lower() or keyword_lower in content.lower():
                results.append({
                    'disease': disease,
                    'content': content
                })
        
        return results
