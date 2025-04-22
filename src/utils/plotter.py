import os
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

        # Highest) R² score in bold
        best_idx = df["Score"].idxmax()
        df["Score"] = df["Score"].astype(str)  # Ensure the column is in string format
        df.loc[best_idx, "Score"] = f"$\\bf{{{df.loc[best_idx, 'Score']}}}$"  # LaTeX bold formatting

        # Plot the table
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 1))
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

            cell.set_height(0.1)
            cell.set_fontsize(12)

        fig.tight_layout()

        if not os.path.exists(os.path.dirname(path_png)):
            os.makedirs(os.path.dirname(path_png))
        plt.savefig(path_png)
        plt.close()