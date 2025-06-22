import os
from inventory_predictor import InventoryPredictor

def main():
    predictor = InventoryPredictor()
    
    print("Order Builder\n")
    
    while True:
        print("Options:")
        print("1.  Train model")
        #print("2. Load trained model")
        print("2.  Make prediction from forecast file")
        print("3.  Exit")
        
        choice = input("Enter choice 1-3:  ")
        
        if choice == '1':
            print("\nTraining Setup")
            report_file = input("Report file path:  ")
            inventory_folder = input("Inventory files folder:  ")
            current_inventory = input("Most recent inventory file:  ")
            
            missing_files = []
            if not os.path.exists(report_file):
                missing_files.append(f"Report file: {report_file}")
            if not os.path.exists(inventory_folder):
                missing_files.append(f"Inventory folder: {inventory_folder}")
            if not os.path.exists(current_inventory):
                missing_files.append(f"Latest inventory file: {current_inventory}")
            
            if missing_files:
                print("\nMissing files:")
                for missing in missing_files:
                    print(f" {missing}")
                continue
            
            print("\nTraining")
            success = predictor.train_model(report_file, inventory_folder, current_inventory)
        
            if success:
                print("\nModel trained")
            else:
                print("\nTraining failed")


        # eli3f choice == '2':1
        #     model_path = input("Model file path (inventory_model.pth): ").strip()
        #     if not model_path:
        #         model_path = "fnn.pth"
            
        #     processor_path = input("Data processor file path (processed_data.pth): ").strip()
        #     if not processor_path:
        #         processor_path = "processed_data.pth"

        #     current_inventory = input("Current inventory file: ").strip()
        #     predictor.load_model(model_path, processor_path, current_inventory)

        #     print("Model loaded! You can now make predictions.")

        elif choice == '2':
            if predictor.neural_network is None:
                print("The model is not trained")
                continue
                
            forecast_file = input("Forecast file path:  ").strip()
            
            if not os.path.exists(forecast_file):
                print(f"No file found: {forecast_file}")
                continue
            
            predictor.predict_inventory_needs(forecast_file)
            
        elif choice == '3':
            print("Exiting")
            break
            
        else:
            print("Enter choice 1-3:  ")

if __name__ == "__main__":
    main()