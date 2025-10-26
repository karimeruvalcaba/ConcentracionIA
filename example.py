import pandas as pd
import nltk
from CounterfactualEvaluator import CounterfactualEvaluator  

def main():
    # Example DataFrame
    df = pd.DataFrame({
        "original_text": [
            "Me gusta el mango",
            "La peor comida son las enchiladas",
        ],
        "cf_text": [
            "Odio el mango",
            "La mejor comida son las enchiladas",
        ],
        "orig_pred": ["Positive", "Negative"],
        "cf_pred": ["Negative", "Positive"],
    })

    df2 = pd.DataFrame({
        "original_text": [
            "Me gusta el mango",
            "La peor comida son las enchiladas",
        ],
        "cf_text": [
            "Amo el mango",
            "La comida mas horrible son las enchiladas",
        ],
        "orig_pred": ["Positive", "Negative"],
        "cf_pred": ["Negative", "Positive"],
    })

    # Create evaluator and compute metrics
    my_model_name = "datificate/gpt2-small-spanish"
    evaluator = CounterfactualEvaluator(model_name=my_model_name)
    results = evaluator.evaluate(df)
    print("\n=== Counterfactual Evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")

    print("--------------- MISMO SENTIMIENTO ---------------")
    evaluator2 = CounterfactualEvaluator(model_name="gpt2")
    results2 = evaluator2.evaluate(df2)
    print("\n=== Counterfactual Evaluation ===")
    for k, v in results2.items():
        print(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
if __name__ == "__main__":
    main()


"""  "cf_text": [
            "Amo el mango",
            # "La comida mas horrible son las enchiladas",
        ], """