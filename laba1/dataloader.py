import pandas as pd
import numpy as np
from pathlib import Path
from config import COLUMN_CONFIG

class DataLoader():

    def __init__(self, file_path):
        self.file_path = file_path

    def extract_excel_data(self) -> dict:
        """
        Extract data from configured columns in Excel file and convert to float type.
        
        Returns:
            dict: Dictionary with column data as pandas Series with float dtype
                  Keys are column identifiers, values are cleaned data
        """
        try:
            # Read Excel file
            df = pd.read_excel(self.file_path)
            
            extracted_data = {}
            
            # Extract data for each configured column
            for col_id, col_config in COLUMN_CONFIG.items():
                col_name = col_config['name']
                col_index = col_config['index']
                
                # Try to access by column name first, then by index
                if col_id in df.columns:
                    col_data = df[col_id]
                else:
                    # Access by index
                    col_data = df.iloc[:, col_index]
                
                # Convert to float, handling any non-numeric values
                col_float = pd.to_numeric(col_data, errors='coerce')
                
                # Remove NaN values
                col_clean = col_float.dropna()
                
                extracted_data[col_id] = {
                    'data': col_clean,
                    'name': col_name,
                    'description': col_config['description']
                }
            
            return extracted_data
            
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found!")
            return None
        except Exception as e:
            print(f"Error reading Excel file: {str(e)}")
            return None