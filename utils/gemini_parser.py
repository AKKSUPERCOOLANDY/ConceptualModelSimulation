import re

class GeminiParser:
    def parse_models(response_text: str):
        models_data = {}
        model_pattern = re.compile(r'(?:\*\*)?(?:MODEL|Model)\s+(\d)(?:\*\*)?[:\s]+(.*?)(?=(?:\*\*)?(?:MODEL|Model)\s+\d|\Z)', 
                                  re.DOTALL | re.IGNORECASE)
        
        matches = model_pattern.findall(response_text)
        
        for model_num_str, description in matches:
            model_num = int(model_num_str)
            # Clean up the description
            description = description.strip()
            
            # Extract model name if it exists
            model_name = "Unnamed Model"
            name_pattern = re.compile(r'^(.*?)(?:\r?\n|\:|$)', re.IGNORECASE)
            name_match = name_pattern.search(description)
            if name_match:
                candidate_name = name_match.group(1).strip()
                # Check if it's actually a name (not too long, not just "Description:" etc.)
                if len(candidate_name.split()) <= 10 and not candidate_name.lower().startswith(('description', 'inference', 'hypothesis', 'approach')):
                    model_name = candidate_name
                    # Remove the name from the description
                    description = description[len(candidate_name):].strip()
                    if description.startswith(':'):
                        description = description[1:].strip()
            
            # Extract hypothesis/inference statement
            hypothesis = ""
            hypothesis_pattern = re.compile(r'(?:hypothesis|inference)(?:[:\s]+)(.*?)(?:\n\n|\n\d|\Z)', re.DOTALL | re.IGNORECASE)
            hypothesis_match = hypothesis_pattern.search(description)
            if hypothesis_match:
                hypothesis = hypothesis_match.group(1).strip()
            else:
                # Try to find a sentence containing hypothesis, inference, or tests
                pattern = r'([^.!?]*(?:hypothesis|inference|test|predict|relationship|correlation|feature)[^.!?]*[.!?])'
                matches = re.findall(pattern, description, re.IGNORECASE)
                if matches:
                    hypothesis = matches[0].strip()
                else:
                    # Just get the first non-empty sentence if no match
                    sentences = re.split(r'[.!?]', description)
                    for sentence in sentences:
                        if sentence.strip():
                            hypothesis = sentence.strip()
                            break
            
            models_data[model_num] = {
                'model_name': model_name,
                'description': description,
                'hypothesis': hypothesis
            }
        
        # If the regex didn't work well, try another approach
        if len(models_data) == 0:
            # Split text by markers like "MODEL X:"
            sections = re.split(r'(?:\*\*)?(?:MODEL|Model)\s+\d+(?:\*\*)?[:\s]+', response_text)
            # First section is usually intro text, remove it
            if len(sections) > 1:
                sections = sections[1:]
            
            # Extract model numbers using simple regex
            model_nums = re.findall(r'(?:\*\*)?(?:MODEL|Model)\s+(\d+)(?:\*\*)?[:\s]+', response_text)
            
            # Match sections with model numbers
            for i, (model_num_str, description) in enumerate(zip(model_nums, sections)):
                if i < len(sections):
                    model_num = int(model_num_str)
                    description = description.strip()
                    
                    # Extract model name
                    model_name = "Unnamed Model"
                    name_pattern = re.compile(r'^(.*?)(?:\r?\n|\:|$)', re.IGNORECASE)
                    name_match = name_pattern.search(description)
                    if name_match:
                        candidate_name = name_match.group(1).strip()
                        # Check if it's actually a name
                        if len(candidate_name.split()) <= 10 and not candidate_name.lower().startswith(('description', 'inference', 'hypothesis', 'approach')):
                            model_name = candidate_name
                            # Remove the name from the description
                            description = description[len(candidate_name):].strip()
                            if description.startswith(':'):
                                description = description[1:].strip()
                    
                    # Extract hypothesis/inference statement
                    hypothesis = ""
                    hypothesis_pattern = re.compile(r'(?:hypothesis|inference)(?:[:\s]+)(.*?)(?:\n\n|\n\d|\Z)', re.DOTALL | re.IGNORECASE)
                    hypothesis_match = hypothesis_pattern.search(description)
                    if hypothesis_match:
                        hypothesis = hypothesis_match.group(1).strip()
                    else:
                        # Try to find a sentence containing hypothesis, inference, or tests
                        pattern = r'([^.!?]*(?:hypothesis|inference|test|predict|relationship|correlation|feature)[^.!?]*[.!?])'
                        matches = re.findall(pattern, description, re.IGNORECASE)
                        if matches:
                            hypothesis = matches[0].strip()
                        else:
                            # Just get the first non-empty sentence if no match
                            sentences = re.split(r'[.!?]', description)
                            for sentence in sentences:
                                if sentence.strip():
                                    hypothesis = sentence.strip()
                                    break
                    
                    models_data[model_num] = {
                        'model_name': model_name,
                        'description': description,
                        'hypothesis': hypothesis
                    }
        
        return models_data