from pydantic import BaseModel


class Outputs(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc_score: float
    mcc: float
