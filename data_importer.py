import pandas as pd
import os
import glob

class DataImporter:
    def __init__(self):
        pass
    
    """
    Loads the sales data report
    """
    def load_sales_data(self, file_path):
        print(f"Loading sales data from {file_path}")
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            
        except Exception as e:
            print(f"Error in loading sales data: {e}")
            return None
        
        df = self.clean_sales_dataframe(df)
        return df
    
    """
    Loads all inventory files in a given folder as separate dataframes
    """
    def load_inventory_files_separately(self, folder_path):
        print("Loading inventory files")
        inventory_files = []
        
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        
        for file_path in excel_files:
            try:                 
                df = self.load_single_inventory_file(file_path)
                    
                if df is not None and not df.empty:
                    filename = os.path.basename(file_path)
                    df['file_date'] = filename
                    df['file_path'] = file_path
                        
                    inventory_files.append(df)
                    print(f"Loaded {filename}")
                else:
                    print(f"Could not load {file_path}")
                        
            except Exception as e:
                print("Error loading files")
                continue
        
        print(f"Loaded {len(inventory_files)} inventory files")
        return inventory_files

    """
    Loads the current inventory file, used for basing current ordering needs
    """
    def load_current_inventory(self, file_path):
        print(f"Loading current inventory file")
        
        if not os.path.exists(file_path):
            print("File not found")
            return None
            
        try:
            df = self.load_single_inventory_file(file_path)
            
            if df is not None and not df.empty:
                filename = os.path.basename(file_path)
                df['file_date'] = filename
                df['file_path'] = file_path
                
                print(f"Loaded {filename}")
                return df
            else:
                print(f"Could not load {filename}")
                return None
                
        except Exception as e:
            print("Error loading current inventory file")
            return None

    """
    Loads forecasting file for bands, weather, events, day, etc
    """
    def load_forecast_file(self, file_path):
        print("Loading forecast file")
        try:
            forecasting_data = []
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) == 6:
                            day_info = {
                                'day': parts[0],
                                'weather': parts[1],
                                'temperature': float(parts[2]),
                                'rain': parts[3].lower() == 'y',
                                'band': parts[4] if parts[4].lower() != 'none' else 'None',
                                'events': parts[5] if parts[5].lower() != 'none' else 'None'
                            }
                            forecasting_data.append(day_info)
                        else:
                            print("Failed to load forecasting file")
                            
                    except Exception as e:
                        print("Error loading forecasting file")
            
            print(f"Successfully loaded forecasting data")
            return forecasting_data
            
        except Exception as e:
            print("Error loading forecasting file in the first try")
            return None
    
    """
    Helper method for loading the inventory folder.  Loads a single inventory file
    """
    def load_single_inventory_file(self, file_path):
        df = None
        try:
            df = pd.read_excel(file_path, sheet_name='Overview', engine='openpyxl')
            
            # Removing rows w/ columns that are null or missing
            if df is not None and not df.empty and len(df.columns) > 5:
                df = df.dropna(how='all')
                df.columns = df.columns.astype(str).str.strip()
                
            return df
                
        except Exception as e:
            print("Error loading sheet")
            return None
        
    """
    Helper method to process the report data
    """
    def clean_sales_dataframe(self, df):
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how='all')
        
        column_mapping = {}

        column_mapping[0] = 'Day of Week'
        column_mapping[1] = 'Date'
        column_mapping[2] = 'Net Sales'
        column_mapping[3] = 'Band'
        column_mapping[4] = 'Private Party'
        column_mapping[5] = 'Avg Temp'
        column_mapping[6] = 'Rain (Y/N)'
        column_mapping[0] = 'Weather'
        column_mapping[0] = 'Events'
        
        df = df.rename(columns=column_mapping)
        
        # Convert to numerical values and fill in missing values if any
        df['Avg Temp'] = pd.to_numeric(df['Avg Temp'], errors='coerce').fillna(70)
        df['Net Sales'] = pd.to_numeric(df['Net Sales'], errors='coerce').fillna(0)
        
        # Filling in missing values if any
        df['Rain (Y/N)'] = df['Rain (Y/N)'].fillna('N')
        df['Weather'] = df['Weather'].fillna('Clear') 
        df['Band'] = df['Band'].fillna('None')
        df['Events'] = df['Events'].fillna('None')
        
        # Date column so the nn can understand what months have better sales
        if 'Date' in df.columns and df['Date'].notna().any():
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month.fillna(1).astype(int)
            df['Week'] = df['Date'].dt.isocalendar().week.fillna(1).astype(int)
        else:
            # Set month to Jan and first week since the bar not open during this time
            df['Month'] = 1
            df['Week'] = 1
        
        print(f"Processed sales")
        return df