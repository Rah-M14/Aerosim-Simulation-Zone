import torch
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from typing import Optional, Dict, Tuple
from .simulator import TrajectorySimulator
from .utils import NavigationUtils

class NavigationTrainer:
    def __init__(
        self,
        net,
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: Optional[str] = None,
        project_name: str = "navigation_training",
        experiment_name: str = "phi_flow_training"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.batch_size = batch_size
        
        # Initialize components
        self.simulator = TrajectorySimulator()
        self.utils = NavigationUtils()
        
        # Training components
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )
        
        # Initialize wandb
        self.config = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "device": self.device,
            "architecture": str(net)
        }
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=self.config
        )

    def train_step(self, initial_pos, goal_pos) -> Tuple[bool, float, Dict]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        try:
            loss = self.simulator.physics_loss(self.net, initial_pos, goal_pos)
            total_loss = loss.sum()  # Sum the loss to get a scalar
            
            if not torch.isfinite(total_loss):
                return False, float('inf'), self._get_empty_metrics()
            
            total_loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            # Check for NaN gradients
            if any(p.grad is not None and not torch.all(torch.isfinite(p.grad)) 
                   for p in self.net.parameters()):
                return False, float(total_loss), self._get_empty_metrics()
            
            self.optimizer.step()
            
            metrics = {
                "gradient_norm": grad_norm.item(),
                "loss_std": 0.0,  # Can be computed if needed
                "successful_trajectories": self._count_successful_trajectories(initial_pos, goal_pos),
                "success_rate": 0.0  # Will be updated later
            }
            metrics["success_rate"] = metrics["successful_trajectories"] / self.batch_size
            
            return True, float(total_loss), metrics
            
        except Exception as e:
            print(f"Error in training step: {e}")
            return False, float('inf'), self._get_empty_metrics()

    def _get_empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            "gradient_norm": 0.0,
            "loss_std": 0.0,
            "successful_trajectories": 0,
            "success_rate": 0.0
        }

    def _count_successful_trajectories(self, initial_pos, goal_pos) -> int:
        """Count number of successful trajectories in batch"""
        with torch.no_grad():
            success_count = 0
            for init, goal in zip(initial_pos, goal_pos):
                final_pos = self.simulator.simulate_trajectory(
                    self.net, init, goal, max_steps=12
                )[-1]
                if torch.norm(final_pos - goal) < 0.1:
                    success_count += 1
            return success_count

    def train(
        self,
        epochs: int = 10000,
        log_interval: int = 500,
        save_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
        visualize_every: int = 1
    ):
        """Training loop with comprehensive logging"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        best_loss = float('inf')
        pbar = tqdm(total=epochs, desc="Training Progress", dynamic_ncols=True)
        
        try:
            for epoch in range(epochs):
                # Generate batch
                initial_pos, goal_pos = self.utils.generate_batch(self.batch_size)
                
                # Training step
                success, loss, metrics = self.train_step(initial_pos, goal_pos)
                if not success:
                    wandb.log({"training_error": 1.0}, step=epoch)
                    continue
                
                # Update learning rate
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'success_rate': f'{metrics["success_rate"]:.2%}',
                    'lr': f'{current_lr:.2e}'
                })
                pbar.update(1)
                
                # Logging
                log_dict = {
                    "loss": loss,
                    "learning_rate": current_lr,
                    **metrics,
                    "epoch": epoch
                }
                
                # Visualization
                if epoch % visualize_every == 0:
                    test_initial, test_goal = self.utils.generate_batch(batch_size=1)
                    fig = self.utils.plot_trajectory(
                        self.net,
                        test_initial.native("samples,vector").squeeze(0),
                        test_goal.native("samples,vector").squeeze(0),
                        show=False
                    )
                    log_dict["trajectory"] = wandb.Image(fig)
                    plt.close(fig)
                
                wandb.log(log_dict, step=epoch)
                
                # Save checkpoints
                if loss < best_loss:
                    best_loss = loss
                    self.utils.save_model(self.net, checkpoint_dir / "best_model.pth")
                    wandb.run.summary["best_loss"] = best_loss
                
                if epoch % save_interval == 0:
                    self.utils.save_model(
                        self.net, 
                        checkpoint_dir / f"checkpoint_{epoch}.pth"
                    )
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving final model...")
            self.utils.save_model(self.net, checkpoint_dir / "interrupted_model.pth")
            
        finally:
            pbar.close()
            wandb.finish()

    def save_model(self, path: str):
        """Save model with metadata"""
        self.utils.save_model(self.net, path)
        wandb.save(path)

    def load_model(self, path: str):
        """Load model with metadata"""
        self.net = self.utils.load_model(self.net, path)
        return self.net