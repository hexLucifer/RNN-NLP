import os
import configparser


def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    while True:
        try:
            choice = int(input("Enter 1 to train, or 2 to generate text: "))
            if choice == 1:
                train_ratio = float(input("Enter the training ratio (0.0 to 1.0): "))
                val_ratio = float(input("Enter the validation ratio (0.0 to 1.0): "))
                test_ratio = 1.0 - train_ratio - val_ratio

                if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
                    print("Invalid ratios. Please ensure they sum up to 1.0.")
                    continue

                config.set("training", "train_ratio", str(train_ratio))
                config.set("training", "val_ratio", str(val_ratio))
                config.set("training", "test_ratio", str(test_ratio))
                with open("config.ini", "w") as configfile:
                    config.write(configfile)

                os.system("python train.py")
                break
            elif choice == 2:
                os.system("python generate.py")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    main()
