from .models import NavigationNet
from .trainer import NavigationTrainer

def main():
    # Initialize network
    net = NavigationNet()
    
    # Initialize trainer
    trainer = NavigationTrainer(
        net=net,
        learning_rate=1e-3,
        batch_size=512,
        project_name="navigation_training",
        experiment_name="phi_flow_training"
    )
    
    # Train the model
    trainer.train(
        epochs=10000,
        log_interval=500,
        save_interval=1000,
        checkpoint_dir="checkpoints",
        visualize_every=1
    )

if __name__ == "__main__":
    main()