import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    
    def __init__(self):
        self.factors = {}
        self.categories = {}
        self.current_inventory = {}
        self.product_list = []
    
    """
    Normalize the factors so its readable in the nn.  Transform is true when learning
    """
    def normalize_factors(self, data, factor, transform=True):
        if transform:
            mean_val = np.mean(data)
            std_val = np.std(data)
            self.factors[factor] = {'mean': mean_val, 'std': std_val}
        else:
            mean_val = self.factors[factor]['mean']
            std_val = self.factors[factor]['std']
        
        if std_val == 0:
            return np.zeros_like(data)
        
        return (data - mean_val) / std_val
    
    """
    Converts the categories into numbers so nn can read 
    """
    def convert_to_categories(self, data, factor, transform=True):
        if transform:
            unique_values = list(set(data))
            # Iterate thru categories
            self.categories[factor] = {val: idx for idx, val in enumerate(unique_values)}
        
        key = self.categories[factor]

        return np.array([key.get(val, 0) for val in data])

    """
    Combines the sales and inventory data based on date
    """
    def merge_sales_with_inventory(self, sales_df, inv_files):
        sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        merged = []
        
        for idx, sales_row in sales_df.iterrows():
            sales_date = sales_row['Date']
            matching_inventory = self.find_matching_inventory(sales_date.date(), inv_files)
            
            if matching_inventory is None:
                continue
                
            for idx, inv_row in matching_inventory.iterrows():
                merged_row = self.merge_files(sales_row, inv_row, sales_date)
                if merged_row:
                    merged.append(merged_row)
        
        return pd.DataFrame(merged)
    
    """
    Helper method to find the inventory file for the matching dates from sales data
    """
    def find_matching_inventory(self, sales_date, inv_files):
        for inv_df in inv_files:
            if inv_df is None or inv_df.empty:
                continue
                
            for idx, inv_row in inv_df.iterrows():

                try:
                    start_date = pd.to_datetime(inv_row.get('Start Date', ''), format='%m/%d/%Y %H:%M', errors='coerce').date()
                    end_date = pd.to_datetime(inv_row.get('End Date', ''), format='%m/%d/%Y %H:%M', errors='coerce').date()
                    
                    # Skip row if no date
                    if pd.isna(start_date) or pd.isna(end_date):
                        continue
                        
                    # Return correct inv if sales date falls w/in start and end
                    if start_date <= sales_date <= end_date:
                        return inv_df
                except:
                    continue

        return None
    
    """
    Merges the rows with corresponding dates
    """
    def merge_files(self, sales_row, inv_row, sales_date):
        try:
            product = str(inv_row.get('Product', '')).strip()
                
            # Set default vals and convert to float
            curr_val = inv_row.get('Curr.', 0)
            usage_val = inv_row.get('Usage.', 0)
            curr_inv = float(curr_val) if pd.notna(curr_val) else 0
            usage = float(usage_val) if pd.notna(usage_val) else 0
            
            return {
                'date': sales_date,
                'day_of_week': sales_row.get('Day of Week'),
                'weather': sales_row.get('Weather', 'Clear'),
                'band': sales_row.get('Band', 'None'),
                'events': sales_row.get('Events', 'None'),
                'temperature': float(sales_row.get('Avg Temp', 70)),
                'rain': sales_row.get('Rain (Y/N)', 'N'),
                'product': product,
                'distributor': str(inv_row.get('Distributor', 'Unknown')).strip(),
                'current_stock': curr_inv,
                'usage': usage,
                'category': str(inv_row.get('Category', '')).lower(),
                'subcategory': str(inv_row.get('Subcategory', '')).lower(),
            }
        
        except Exception:
            return None
    
    """
    Helper method that merges the data an categorizes and normalize input factors
    """
    def prepare_training_data(self, sales_df, inventory_files_data=None):
        merged_df = self.merge_sales_with_inventory(sales_df, inventory_files_data)
        
        self.product_list = sorted(merged_df['product'].unique().tolist())
        
        X = self.get_factors(merged_df)
        y = self.normalize_factors(np.array(merged_df['usage'].fillna(0)), 'usage', transform=True).reshape(-1, 1)
        
        return X, y
    
    """
    Converts the factors into categories based on the key.  Sets default values for the ones that are missing
    """
    def get_factors(self, merged_df):
        day = self.convert_to_categories(merged_df['day_of_week'].fillna('Monday'), 'day_of_week', transform = True)
        weather = self.convert_to_categories(merged_df['weather'].fillna('Clear'), 'weather', transform = True)
        band = self.convert_to_categories(merged_df['band'].fillna('None'), 'band', transform = True)
        events = self.convert_to_categories(merged_df['events'].fillna('None'), 'events', transform = True)
        product = self.convert_to_categories(merged_df['product'].fillna('Unknown'), 'product', transform = True)
        distributor = self.convert_to_categories(merged_df['distributor'].fillna('Unknown'), 'distributor', transform = True)
        category = self.convert_to_categories(merged_df['category'].fillna(''), 'category', transform = True)
        inv = self.normalize_factors(np.array(merged_df['current_stock'].fillna(0)), 'current_stock', transform = True)
        temp = self.normalize_factors(np.array(merged_df['temperature'].fillna(70)), 'temperature', transform = True)
        rain = np.array([1 if str(r).upper() == 'Y' else 0 for r in merged_df['rain'].fillna('N')])
        
        return np.column_stack([
            day, weather, band, events,
            product, distributor, category,
            temp, inv, rain
        ])
    
    """
    Processes and stores current inventory so its used in predictions
    """
    def analyze_current_inventory(self, latest_inventory_df):      
        columns = self.find_inv_cols(latest_inventory_df)
        
        if not columns['current_col'] or not columns['usage_col']:
            return
        
        self.current_inventory = {}
        
        try:
            inv = latest_inventory_df.copy()
            inv[columns['current_col']] = pd.to_numeric(inv[columns['current_col']], errors='coerce').fillna(0)

            # Fills in default usage to 1 (breaks when 0)
            inv[columns['usage_col']] = pd.to_numeric(inv[columns['usage_col']], errors='coerce').fillna(1)
            inv[columns['usage_col']] = inv[columns['usage_col']].abs()
            
            for idx, row in inv.iterrows():
                self.inv_row(row, columns)
                        
        except Exception:
            self.current_inventory = {}

    """
    Finds the colum names in the inventory files
    """
    def find_inv_cols(self, df):
        columns = {'product_col': None, 'distributor_col': None, 'current_col': None, 'usage_col': None}
        
        for column in df.columns:
            col = str(column).strip().lower()
            
            if col == 'product':
                columns['product_col'] = column
            elif col == 'distributor':
                columns['distributor_col'] = column
            elif col == 'curr.':
                columns['current_col'] = column
            elif col == 'usage.':
                columns['usage_col'] = column
        
        return columns
    
    """
    Sets the rows of the inventory
    """
    def inv_row(self, row, columns):
        try:
            product = str(row[columns['product_col']]).strip()
            dist = row[columns['distributor_col']] if columns['distributor_col'] else None
            curr = row[columns['current_col']]
            usage = row[columns['usage_col']]

            # Set default vals for rows
            if columns['distributor_col']:
                distributor = str(dist).strip()
            else:
                distributor = 'Unknown'

            current = float(curr) if pd.notna(curr) else 0

            if pd.notna(usage) and usage > 0:
                usage = float(usage)
            else:
                usage = 1

            if distributor not in self.current_inventory:
                self.current_inventory[distributor] = {}
            
            self.current_inventory[distributor][product] = {
                'current_stock': current,
                'weekly_usage': usage,
                'usage_rate': usage / 7,
                'category': str(row.get('Category', '')).lower(),
                'subcategory': str(row.get('Subcategory', '')).lower(),
            }
        except Exception:
            pass
    
    """
    Similar to get factors by converting the factors into categories based on the key.  Sets default values for the ones that are missing.
    Flags the key to not train the nn
    """
    def prepare_prediction_features(self, forecast_data):
        all_features = []
        
        for day_info in forecast_data:
            for product in self.product_list:
                current_stock_info = self.get_curr_inv(product)
                
                day = self.convert_to_categories([day_info['day']], 'day_of_week', transform = False)[0]
                weather = self.convert_to_categories([day_info['weather']], 'weather', transform = False)[0]
                band = self.convert_to_categories([day_info['band']], 'band', transform = False)[0]
                events = self.convert_to_categories([day_info['events']], 'events', transform = False)[0]
                products = self.convert_to_categories([product], 'product', transform = False)[0]
                distributor = self.convert_to_categories([current_stock_info['distributor']], 'distributor', transform = False)[0]
                category = self.convert_to_categories([current_stock_info['category']], 'category', transform = False)[0]
                stock = self.normalize_factors([current_stock_info['current_stock']], 'current_stock', transform = False)[0]
                temp = self.normalize_factors([day_info['temperature']], 'temperature', transform = False)[0]
                rain = 1 if day_info['rain'] else 0
                
                # Second product is a string
                day_features = [
                    day, weather, band, events,
                    products, distributor, category,
                    temp, rain, stock, product
                ]

                all_features.append(day_features)
        
        return all_features
    
    """
    Helper to get the current inventory for a specifc product
    """
    def get_curr_inv(self, product):
        for distributor, products in self.current_inventory.items():
            if product in products:
                info = products[product].copy()
                info['distributor'] = distributor
                return info
        
        return {'distributor': 'Unknown', 'current_stock' : 0, 'category': ''}