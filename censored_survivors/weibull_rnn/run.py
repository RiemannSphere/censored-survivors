try:
    from .models import (
        WeibullRNNData,
        WeibullRNNInput,
        WeibullRNNOutput,
        WeibullRNNResult,
        WeibullRNNPrediction,
        WeibullParameters
    )
except ImportError:
    from models import (
        WeibullRNNData,
        WeibullRNNInput,
        WeibullRNNOutput,
        WeibullRNNResult,
        WeibullRNNPrediction,
        WeibullParameters
    ) 