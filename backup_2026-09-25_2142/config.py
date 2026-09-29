# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

GOOGLE_TRACE_PATH = RAW_DIR / "borg_traces_data.csv"
PROCESSED_GOOGLE = PROCESSED_DIR / "google_processed.csv"
PROCESSED_ALIBABA = PROCESSED_DIR / "alibaba_processed.csv"

# Alibaba trace paths
ALIBABA_CONTAINER_META = RAW_DIR / "container_meta.csv"
ALIBABA_MACHINE_META = RAW_DIR / "machine_meta.csv"
ALIBABA_CONTAINER_USAGE = RAW_DIR / "container_usage.csv"

# container_meta: header-less, 8 columns (confirmed)
ALIBABA_CONTAINER_META_COLS = [
    "container_id", "machine_id", "time_stamp", "app_du",
    "status", "cpu_request", "cpu_limit", "mem_request",
]

# machine_meta: header-less, 7 columns (corrected)
ALIBABA_MACHINE_META_COLS = [
    "machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
    "cpu_num", "mem_size", "status",
]

# Environment
NUM_VMS = 50
MAX_QUEUE = 100
EPISODE_LENGTH = 2000
MAX_STEPS_PER_EPISODE = 3000

# DQN
LEARNING_RATE = 5e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
REPLAY_BUFFER = 100_000
HIDDEN_DIM = 96

# Training
NUM_EPISODES = 300
SEED = 42
LOG_INTERVAL = 10

# Outputs
MODEL_PATH = RESULTS_DIR / "dqn_google.pt"
TRAIN_LOG_PATH = RESULTS_DIR / "training_log.csv"
EVAL_LOG_PATH = RESULTS_DIR / "evaluation_log.csv"