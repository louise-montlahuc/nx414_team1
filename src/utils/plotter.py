import os
import re
import pandas as pd
import matplotlib.pyplot as plt

class Plotter():
    @staticmethod
    def save_corr_plot(data, title, path):
        plt.hist(data, bins=20, edgecolor='k', alpha=0.7)
        plt.xlabel('Pearson correlation coefficient')
        plt.ylabel('# neurons')
        plt.title(title)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        plt.savefig(path)
        plt.close()
    
    @staticmethod
    def update_r2_score_csv(new_scores: dict, path: str):
        """
        Update or create a CSV file to store R² scores per model per layer.

        Args:
            model_name (str): Name of the model.
            r2_scores (dict): Mapping of layer names to R² scores.
            save_path (str): Path to the CSV file.
        """
        # Update csv if it already exists
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            r2_scores = df["Score"].to_dict()
        # Create csv if it doesn't exist
        else:
            r2_scores = {}

        r2_scores.update(new_scores)

        df = pd.DataFrame(list(r2_scores.items()), columns=["Model", "Score"])
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path, index=False)

        print("Updated scores:", r2_scores)

    @staticmethod
    def save_r2_table(path_csv, path_png):
        """
        Save R² scores for models and generate a summary table image.
        
        Parameters:
            model_scores (dict): Dictionary with model names as keys and R² scores as values.
            save_path (str): File path to save the output table image.
        """
        # Use csv if it already exists
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv)
        # Create csv if it doesn't exist
        else:
            raise FileNotFoundError(f"The file '{path_csv}' was not found.")

        # Fill NaN values with 0 and round R² scores
        df["Score"] = pd.to_numeric(df["Score"], errors='coerce').fillna(0).round(4)

        # Sort by Score descending
        df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

        # Highest) R² score in bold
        best_idx = df["Score"].idxmax()
        df["Score"] = df["Score"].astype(str)  # Ensure the column is in string format
        df.loc[best_idx, "Score"] = f"$\\bf{{{df.loc[best_idx, 'Score']}}}$"  # LaTeX bold formatting

        # Arrange the name
        names = df["Model"].tolist()
        for i in range(len(names)):
            names[i] = names[i].replace("_", " ") 
            names[i] = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', names[i]) # Separate letters and digits
            names[i] = re.sub(r'\b(all)\b', '', names[i], flags=re.IGNORECASE) # Remove 'all'
            names[i] = re.sub(r'\s+', ' ', names[i]).strip()
        df["Model"] = names

    
        # Plot the table
        row_height = 0.4  
        fig_height = len(df) * row_height + 1  
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        # Style header and model column
        header_color = "#d1e7ff"
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(header_color)
            elif col == 0:
                cell.set_facecolor(header_color)
            cell.set_height(0.05)
            cell.set_width(0.9)
            cell.set_fontsize(12)

        plt.tight_layout()

        if not os.path.exists(os.path.dirname(path_png)):
            os.makedirs(os.path.dirname(path_png))
        plt.savefig(path_png, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_training_history(train_losses, valid_losses, r2_scores, lrs, epochs, model_name, layer_name):
        """
        Plot the training history of the model.

        Args:
            train_losses (list): List of training losses.
            valid_losses (list): List of validation losses.
            r2_scores (list): List of R² scores.
            lrs (list): List of learning rates.
            epochs (int): Number of epochs.
            model_name (str): Name of the model.
            layer_name (str): Name of the layer.
        """
        nb_plots = 2 if len(r2_scores) == 0 else 3

        plt.figure(figsize=(6 * nb_plots, 5))

        # Plot training and validation loss
        plt.subplot(1, nb_plots, 1)
        plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='blue')
        plt.plot(range(1, epochs + 1), valid_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_name} - Training and Validation Loss')
        plt.legend()

        # Plot learning rate
        plt.subplot(1, nb_plots, 2)
        plt.plot(range(1, epochs + 1), lrs, label='Learning Rate', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title(f'{model_name} - Learning Rate Schedule')
        plt.legend()

        if len(r2_scores) > 0:
            # Plot R² scores
            plt.subplot(1, 3, 3)
            plt.plot(range(1, epochs + 1), r2_scores, label=r'$R^2$', color='green')
            plt.xlabel('Epochs')
            plt.ylabel(r'$R^2$')
            plt.title(f'{model_name} - R² Score')
            plt.legend()

        if not os.path.exists(os.path.dirname('./saved/plots/')):
            os.makedirs(os.path.dirname('./saved/plots/'))
        
        plt.savefig(f'./saved/plots/{model_name}_{layer_name}_training_history_{epochs}ep.png')
        plt.close()