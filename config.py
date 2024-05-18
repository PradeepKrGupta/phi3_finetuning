'''
Data Configuration
'''
DATASET_PATH = "./phiDataset.csv"
DATASET_TEXT_FIELD = "prompt_response"

'''
Model Configuration
'''
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TRUST_REMOTE_CODE = True
ENABLE_MODEL_CONFIG_CACHE = False

'''
Quantization Configuration
'''
ENABLE_4BIT = True
QUANTIZATION_TYPE = "nf4"

'''
Adapter Configuration
'''
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_RANK = 64
TASK_TYPE = "CAUSAL_LM"

'''
Model Training Configuration
'''
MODEL_OUTPUT_DIR = "results/"
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 6
OPTIM = "paged_adamw_32bit"
SAVE_STEPS = 100
LOGGING_STEPS = 10
LEARNING_RATE = 2e-4
MAX_GRAD_NORM = 0.3
MAX_STEPS = 700
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "constant"
ENABLE_FP_16 = True
ENABLE_GRADIENT_CHECKPOINTING=True

'''
Model Trainer Configuration
'''
MAX_SEQ_LENGTH = 512

'''
Inference Configuration
'''
TASK = "text_generation"
