import numpy as np
from datetime import datetime
from data_importer import DataImporter
from data_processor import DataProcessor
from neural_network import NeuralNetwork
import pickle

class InventoryPredictor:
    
    def __init__(self):
        self.data_importer = DataImporter()
        self.data_processor = DataProcessor()
        self.neural_network = None
        
    def train_model(self, sales_file, inventory_folder, latest_inventory_file):
        print("Training Model")
        
        sales_data = self.data_importer.load_sales_data(sales_file)
        
        inventory_files_list = self.data_importer.load_inventory_files_separately(inventory_folder)
        
        current_inventory = self.data_importer.load_current_inventory(latest_inventory_file)
        
        X, y = self.data_processor.prepare_training_data(sales_data, inventory_files_list)
        
        print(f"Training shape:  {X.shape}")
        print(f"Target shape:  {y.shape}")

        self.data_processor.analyze_current_inventory(current_inventory)
        
        if X is not None and y is not None and len(X) > 0:
            input_size = X.shape[1]
            hidden_sizes = [32, 16, 8]
            output_size = 1
            
            self.neural_network = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate=0.001)
            
            print("Training neural network")
            loss = self.neural_network.train(X, y, epochs=500)
            print("Training completed!")
            
            # model_path = "fnn.pth"
            # processor_path = "processed_data.pth"
            # self.neural_network.trainer.save_model(model_path)
            
            # with open(processor_path, 'wb') as f:
            #     pickle.dump({
            #         'encoders': self.data_processor.encoders,
            #         'feature_stats': self.data_processor.feature_stats,
            #         'product_list': self.data_processor.product_list,
            #         'current_inventory': self.data_processor.current_inventory
            #     }, f)

            return True
        else:
            print("No training data available!")
            return False
        
    # def load_model(self, model_path, processor_path, latest_inventory_file):
    #     input_size = 10
    #     hidden_sizes = [32, 16, 8]
    #     output_size = 1

    #     self.neural_network = NeuralNetwork(input_size, hidden_sizes, output_size)

    #     try:
    #         self.neural_network.trainer.load_model(model_path)
    #         print("Model loaded successfully")

    #         with open(processor_path, 'rb') as f:
    #             processed_data = pickle.load(f)

    #         self.data_processor.encoders = processed_data['encoders']
    #         self.data_processor.feature_stats = processed_data['feature_stats']
    #         self.data_processor.product_list = processed_data.get('product_list', [])

    #         if 'current_inventory' in processed_data:
    #             self.data_processor.current_inventory = processed_data['current_inventory']

    #         # Update with latest inventory
    #         current_inventory = self.data_importer.load_current_inventory(latest_inventory_file)
    #         if current_inventory is not None:
    #             self.data_processor.analyze_current_inventory(current_inventory)

    #         return True
    #     except Exception as e:
    #         print(f"Cannot load model: {e}")
    #         return False
    
    """
    Predicts the potential inventory needs
    """
    def predict_inventory_needs(self, weather_file):
        if self.neural_network is None:
            print("The model has not been trained")
            return None
            
        print("Predicting Inventory Needs")
        
        weather_data = self.data_importer.load_forecast_file(weather_file)
        
        prediction = self.data_processor.prepare_prediction_features(weather_data)
        
        # Prediction factors specific to each produc
        products_factors = {}
        for factors in prediction:
            product_name = factors[-1]

            fact = np.array(factors[:-1], dtype=float)
            
            if product_name not in products_factors:
                # Set to nothing if not found in prev files
                products_factors[product_name] = []
            products_factors[product_name].append(fact)
        
        product_predictions = {}
        for product, features in products_factors.items():
            if len(features) == 0:
                continue
                
            X_prod = np.array(features)

            daily_predictions = self.neural_network.make_prediction(X_prod)
            
            # If no product, then mean and std set to 1 for denormalizing
            usage = self.data_processor.factors.get('usage', {'mean': 1, 'std': 1})

            # Denormalize preds w/ val * standard dev + mean
            daily_usage = daily_predictions * usage['std'] + usage['mean']
            daily_usage = np.maximum(daily_usage, 0)
            
            weekly_usage = np.sum(daily_usage)
            
            product_predictions[product] = {
                'weekly_total': weekly_usage,
                'confidence': self.calculate_confidence(daily_usage)
            }
        
        self.generate_recommendations(product_predictions)
        
        return product_predictions
    
    """
    Confidence calculation for how likely it is needed
    """
    def calculate_confidence(self, daily_usage):        
        std = np.std(daily_usage)
        mean = np.mean(daily_usage)
        
        if mean > 0:
            cv = std / mean
            confidence = max(0.3, min(0.95, 1 - cv))
        else:
            # 50/50 on if it should be ordered
            confidence = 0.5
        
        return confidence
    
    """
    Create the recommendations
    """
    def generate_recommendations(self, product_predictions):
        print("\nSuggested Order:\n")
        
        current_inventory = self.data_processor.current_inventory
        
        # Curr inv since it still has all prods even if 0 in stock
        for distributor, products in current_inventory.items():
            print("\n" + str(distributor.upper()))
            
            orders = []
            
            for product, info in products.items():
                current_stock = info['current_stock']
                
                # Skip products that are not needed
                if self.is_necessary(product, info):
                    continue
                
                if product not in product_predictions:
                    continue
                
                predicted_usage = product_predictions[product]['weekly_total']
                confidence = product_predictions[product]['confidence']
                

                target_weeks = 2
                target_stock = predicted_usage * target_weeks
                order_quantity = max(1, int(target_stock - current_stock))
                
                # Round orders so it matches normal orders
                order_quantity = self.round_orders(product, info, order_quantity)
                
                orders.append({
                    'product': product,
                    'current': current_stock,
                    'order': order_quantity,
                    'confidence': confidence
                })
            
            if orders:
                for order in orders:
                    print(f"{order['product']} Current: {order['current']} "
                          f"Needed: {order['order']} "
                          f"Confidence: {order['confidence']:.1%}")
            else:
                print("All products stocked, no order needed")
    
    """
    Rounds orders so that beers are in sets of 24, wines max at 36, liquor in cases of 12, kegs max of 2
    """
    def round_orders(self, product, info, order_quantity):
        category = info.get('category', '').lower()
        subcategory = info.get('subcategory', '').lower()
        product_lower = product.lower()
        
        keg = ('draught' in subcategory)
        beer = ('beer' in category)
        bottle_can = ('bottle' in subcategory or 'can' in subcategory)
        mixer = bottle_can and not beer
        
        if keg:
            order_quantity = max(1, min(order_quantity, 2))

        elif mixer:
            order_quantity = max(1, int((order_quantity + 2) // 6) * 6)
            order_quantity = max(1, min(order_quantity, 12))

        elif beer and bottle_can:
            order_quantity = max(24, int((order_quantity + 11) // 24) * 24)
            order_quantity = min(order_quantity, 144)

        elif 'wine' in category:
            order_quantity = max(6, int((order_quantity + 2) // 6) * 6)
            order_quantity = min(order_quantity, 36)

        elif ('spirit' in category):
            order_quantity = max(1, int(order_quantity))
            order_quantity = min(order_quantity, 12)

        else:
            order_quantity = max(1, int(order_quantity))
            order_quantity = min(order_quantity, 6)
        
        return order_quantity
    
    """
    Check to see if a specific product is needed
    """
    def is_necessary(self, product, info):
        if info['weekly_usage'] < 0.5:
            return True
        
        # Not necessary if more than 2weeks of supply
        if info['weekly_usage'] > 0:
            weeks_of_inv = info['current_stock'] / info['weekly_usage']
        else:
            return False
        if weeks_of_inv > 2:
            return True
        return False