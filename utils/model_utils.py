class ModelUtils:
    def sort_models(models: list):
        sorted_all_models = sorted(models, key=lambda x: (x['metrics']['accuracy_score'] if x and 'metrics' in x and 'accuracy_score' in x['metrics'] else 0, -(x['metrics']['false_negatives'] if x and 'metrics' in x and 'false_negatives' in x['metrics'] else float('inf'))), reverse=True)
        return sorted_all_models