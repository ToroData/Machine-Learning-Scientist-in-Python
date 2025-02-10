import wandb
from project import main

if __name__ == "__main__":
    wandb.init(
        project="predictive-modeling-for-agriculture-datacamp",
    )
    main()