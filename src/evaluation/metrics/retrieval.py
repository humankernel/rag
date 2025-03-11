
class PrecisionRecallF1:
    
    def _align_claims(self, generated: str, reference: str) -> tuple[set[str], set[str]]:
        # Simple lexical alignment for demonstration
        gen_claims = set(generated.lower().split('. '))
        ref_claims = set(reference.lower().split('. '))
        return gen_claims, ref_claims

    def __call__(self, actual: str, reference: str):
        gen, ref = self._align_claims(actual, reference)
        tp = len(gen & ref)
        fp = len(gen - ref)
        fn = len(ref - gen)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) >0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}