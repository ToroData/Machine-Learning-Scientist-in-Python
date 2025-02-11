import wandb
import project

if __name__ == "__main__":
    wandb.init(
        project="Clustering-Antarctic-Penguin-Species"
    )
    penguins_df = project.read_csv("data/penguins.csv")
    X = project.scale(penguins_df)
    project.elbow_plot(X, k_range=[1, 10])
    project.kmeans(X)
