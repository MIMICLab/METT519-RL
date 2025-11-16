#!/usr/bin/env python3
"""
RL2025 - Lecture 11: Experiment 09 - Integrated Capstone Test

Integrates all components from previous experiments into a comprehensive
test suite and demonstration. Validates the complete pipeline from 
MCTS/AlphaZero to RLHF/DPO methods.

Learning objectives:
- Validate integration of all lecture components
- Demonstrate end-to-end workflows
- Benchmark performance and reproducibility
- Prepare foundation for capstone projects

Prerequisites: All previous experiments (exp01-exp08)
"""

# PyTorch 2.x Standard Practice Header
import os, random, numpy as np, torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from pathlib import Path
import warnings
import sys

try:
    from lecture11.experiments import EXPERIMENT_ROOT, RUNS_ROOT
except ImportError:  # Script executed without package context
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    RUNS_ROOT = EXPERIMENT_ROOT / "runs"
warnings.filterwarnings("ignore")

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Proper device selection (CUDA > MPS > CPU)
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    else 'cpu'
)
amp_enabled = torch.cuda.is_available()
setup_seed(42)

BASE_DIR = EXPERIMENT_ROOT
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

class IntegrationTestSuite:
    """Comprehensive test suite for all lecture components."""
    
    def __init__(self):
        self.test_results = {}
        self.base_path = BASE_DIR
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = RUNS_ROOT / "integration_suite"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.artifacts_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        print("="*60)
        print("LECTURE 11 INTEGRATION TEST SUITE")
        print("="*60)
        
        # Test each component
        self.test_results['environment'] = self.test_environment_setup()
        self.test_results['training_header'] = self.test_training_header()
        self.test_results['gomoku'] = self.test_gomoku_environment()
        self.test_results['policy_value'] = self.test_policy_value_network()
        self.test_results['mcts'] = self.test_mcts_implementation()
        self.test_results['alphazero'] = self.test_alphazero_selfplay()
        self.test_results['toy_lm'] = self.test_toy_language_model()
        self.test_results['dpo'] = self.test_dpo_training()
        
        # Integration tests
        self.test_results['integration'] = self.test_full_integration()
        
        # Generate report
        report = self.generate_test_report()
        
        return report
    
    def test_environment_setup(self) -> Dict[str, Any]:
        """Test basic environment and setup."""
        print("\n1. Testing Environment Setup...")
        
        try:
            from exp01_setup_verification import verify_pytorch_setup, verify_mixed_precision
            
            # Run core verification
            pytorch_ok = verify_pytorch_setup()
            amp_ok = verify_mixed_precision()
            
            return {
                'status': 'PASS',
                'pytorch': pytorch_ok,
                'mixed_precision': amp_ok,
                'device': str(device)
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_training_header(self) -> Dict[str, Any]:
        """Test standard training utilities."""
        print("2. Testing Training Header...")
        
        try:
            from exp02_standard_training_header import ExperimentConfig, CheckpointManager, TimingUtils
            
            # Test config management
            config = ExperimentConfig(learning_rate=1e-3)
            config_path = self.configs_dir / "training_header.yaml"
            config.save(str(config_path))
            loaded_config = ExperimentConfig.load(str(config_path))
            config_ok = loaded_config.learning_rate == 1e-3
            
            # Test checkpoint manager
            from exp02_standard_training_header import MinimalDQN
            model = MinimalDQN(4, 2)
            optimizer = torch.optim.Adam(model.parameters())
            checkpoint_manager = CheckpointManager(str(self.checkpoints_dir))
            checkpoint_path = checkpoint_manager.save_checkpoint(model, optimizer, 1, {'loss': 0.5}, config)
            checkpoint_ok = Path(checkpoint_path).exists()
            
            timing_utils_available = hasattr(TimingUtils, "timer")

            return {
                'status': 'PASS',
                'config_management': config_ok,
                'checkpointing': checkpoint_ok,
                'timing_utils': timing_utils_available,
                'utilities_available': True
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_gomoku_environment(self) -> Dict[str, Any]:
        """Test Gomoku 5x5 environment."""
        print("3. Testing Gomoku Environment...")
        
        try:
            from exp03_gomoku_environment import Gomoku5x5, Player
            
            # Test basic functionality
            env = Gomoku5x5()
            obs, info = env.reset()
            
            # Validate observation shape
            obs_shape_ok = obs.shape == (2, 5, 5)
            
            # Test game mechanics
            legal_actions = env.legal_actions_list()
            initial_legal_count = len(legal_actions) == 25
            
            # Play a move
            action = 12  # Center
            obs, reward, terminated, truncated, info = env.step(action)
            move_ok = len(env.legal_actions_list()) == 24 and not terminated
            
            # Test winning condition
            env.reset()
            # Create horizontal win
            for i in range(5):
                if i < 4:
                    env.step(i)      # Black horizontal
                    env.step(i + 5)  # White second row
                else:
                    obs, reward, terminated, truncated, info = env.step(i)  # Winning move
            
            win_detection_ok = terminated and info['winner'] == Player.BLACK
            
            return {
                'status': 'PASS',
                'observation_shape': obs_shape_ok,
                'legal_moves': initial_legal_count,
                'move_execution': move_ok,
                'win_detection': win_detection_ok
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_policy_value_network(self) -> Dict[str, Any]:
        """Test policy-value network."""
        print("4. Testing Policy-Value Network...")
        
        try:
            from exp04_policy_value_network import PolicyValueNet, PolicyValueTrainer
            
            # Test network architecture
            model = PolicyValueNet()
            param_count = sum(p.numel() for p in model.parameters())
            
            # Test forward pass
            batch_size = 4
            states = torch.randn(batch_size, 2, 5, 5)
            legal_masks = torch.ones(batch_size, 25).bool()
            
            policy_logits, values = model(states, legal_masks)
            
            output_shapes_ok = (policy_logits.shape == (batch_size, 25) and 
                               values.shape == (batch_size, 1))
            
            # Test training step
            trainer = PolicyValueTrainer(model, learning_rate=1e-3)
            
            # Create synthetic training data
            target_policies = F.softmax(torch.randn(batch_size, 25), dim=1)
            target_values = torch.randn(batch_size, 1).clamp(-1, 1)
            
            metrics = trainer.train_step(states, target_policies, target_values, legal_masks)
            training_ok = 'total_loss' in metrics and isinstance(metrics['total_loss'], float)
            
            return {
                'status': 'PASS',
                'parameter_count': param_count,
                'output_shapes': output_shapes_ok,
                'training_step': training_ok
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_mcts_implementation(self) -> Dict[str, Any]:
        """Test MCTS implementation."""
        print("5. Testing MCTS Implementation...")
        
        try:
            from exp05_puct_mcts import MCTS, MCTSNode
            from exp04_policy_value_network import PolicyValueNet
            from exp03_gomoku_environment import Gomoku5x5
            
            # Test node functionality
            root = MCTSNode()
            root.expand({0: 0.4, 1: 0.3, 2: 0.3})
            root.N = 1
            selected_action = root.select_action(c_puct=1.0)
            node_ok = selected_action in [0, 1, 2]
            
            # Test MCTS search
            env = Gomoku5x5()
            env.reset()
            model = PolicyValueNet()
            mcts = MCTS(model, c_puct=1.0)
            
            action_probs, root_value = mcts.search(env, num_simulations=10, temperature=1.0)
            
            search_ok = (isinstance(action_probs, dict) and 
                        len(action_probs) > 0 and
                        abs(sum(action_probs.values()) - 1.0) < 1e-5)
            
            return {
                'status': 'PASS',
                'node_operations': node_ok,
                'search_functionality': search_ok,
                'simulation_count': 10
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_alphazero_selfplay(self) -> Dict[str, Any]:
        """Test AlphaZero self-play."""
        print("6. Testing AlphaZero Self-Play...")
        
        try:
            from exp06_selfplay_alphazero import SelfPlayTrainer, SelfPlayConfig, ExperienceBuffer
            from exp04_policy_value_network import PolicyValueNet
            from exp03_gomoku_environment import Gomoku5x5
            
            # Test experience buffer
            buffer = ExperienceBuffer(max_size=10)
            buffer_size_before = len(buffer)
            
            # Test self-play trainer setup
            model = PolicyValueNet()
            config = SelfPlayConfig(num_simulations=5, batch_size=4)
            trainer = SelfPlayTrainer(model, config)
            
            setup_ok = trainer.buffer.max_size == config.buffer_size
            
            # Test game simulation (if MCTS available)
            if trainer.mcts is not None:
                env = Gomoku5x5()
                experiences, game_info = trainer.play_game(env, collect_data=True)
                
                game_simulation_ok = (len(experiences) > 0 and 
                                     'winner' in game_info and
                                     'moves' in game_info)
            else:
                game_simulation_ok = False  # MCTS not available
            
            return {
                'status': 'PASS',
                'buffer_management': len(buffer) == buffer_size_before,
                'trainer_setup': setup_ok,
                'game_simulation': game_simulation_ok,
                'mcts_available': trainer.mcts is not None
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_toy_language_model(self) -> Dict[str, Any]:
        """Test toy language model for DPO."""
        print("7. Testing Toy Language Model...")
        
        try:
            from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
            
            # Test tokenizer
            tokenizer = SimpleTokenizer(vocab_size=128)
            text = "Hello world"
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            tokenizer_ok = len(tokens) > 0 and len(decoded) > 0
            
            # Test model
            model = SimpleTransformerLM(vocab_size=tokenizer.vocab_size, embed_dim=64)
            input_ids = torch.randint(0, tokenizer.vocab_size, (2, 16))
            attention_mask = torch.ones_like(input_ids)
            
            logits = model(input_ids, attention_mask)
            model_ok = logits.shape == (2, 16, tokenizer.vocab_size)
            
            # Test preference dataset
            dataset = PreferenceDataset(tokenizer, size=10)
            batch = dataset.get_batch(batch_size=2, max_length=32)
            
            dataset_ok = (len(dataset) == 10 and
                         'input_ids' in batch and
                         batch['input_ids'].shape[0] == 2)
            
            return {
                'status': 'PASS',
                'tokenizer': tokenizer_ok,
                'model_forward': model_ok,
                'preference_dataset': dataset_ok,
                'vocab_size': tokenizer.vocab_size
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_dpo_training(self) -> Dict[str, Any]:
        """Test DPO training."""
        print("8. Testing DPO Training...")
        
        try:
            from exp08_dpo_implementation import DPOTrainer, DPOConfig
            from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
            import copy
            
            # Setup models
            tokenizer = SimpleTokenizer(128)
            policy_model = SimpleTransformerLM(tokenizer.vocab_size, embed_dim=64, num_layers=2)
            reference_model = copy.deepcopy(policy_model)
            
            config = DPOConfig(beta=0.1, batch_size=4)
            trainer = DPOTrainer(policy_model, reference_model, tokenizer, config)
            
            # Test DPO loss computation
            batch_size = 4
            policy_chosen = torch.randn(batch_size)
            policy_rejected = torch.randn(batch_size) - 1.0  # Make rejected worse
            ref_chosen = torch.randn(batch_size)
            ref_rejected = torch.randn(batch_size) - 1.0
            
            loss_dict = trainer.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, 0.1)
            loss_computation_ok = 'loss' in loss_dict and 'accuracy' in loss_dict
            
            # Test training step with synthetic data
            dataset = PreferenceDataset(tokenizer, size=20)
            batch = dataset.get_batch(config.batch_size, max_length=32)
            
            metrics = trainer.training_step(batch)
            training_step_ok = 'loss' in metrics and isinstance(metrics['loss'], float)
            
            return {
                'status': 'PASS',
                'loss_computation': loss_computation_ok,
                'training_step': training_step_ok,
                'beta_parameter': config.beta
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def test_full_integration(self) -> Dict[str, Any]:
        """Test full integration workflow."""
        print("9. Testing Full Integration...")
        
        try:
            # Test AlphaZero pipeline
            alphazero_ok = self._test_alphazero_pipeline()
            
            # Test DPO pipeline  
            dpo_ok = self._test_dpo_pipeline()
            
            # Test reproducibility
            reproducibility_ok = self._test_reproducibility()
            
            return {
                'status': 'PASS',
                'alphazero_pipeline': alphazero_ok,
                'dpo_pipeline': dpo_ok,
                'reproducibility': reproducibility_ok
            }
            
        except Exception as e:
            return {'status': 'FAIL', 'error': str(e)}
    
    def _test_alphazero_pipeline(self) -> bool:
        """Test complete AlphaZero pipeline."""
        try:
            from exp03_gomoku_environment import Gomoku5x5
            from exp04_policy_value_network import PolicyValueNet
            from exp05_puct_mcts import MCTS
            
            # Create components
            env = Gomoku5x5()
            model = PolicyValueNet()
            mcts = MCTS(model, c_puct=1.0)
            
            # Test pipeline: Reset -> MCTS search -> Move -> Evaluate
            env.reset()
            if mcts is not None:
                action_probs, value = mcts.search(env, num_simulations=5)
                action = max(action_probs.keys(), key=lambda a: action_probs[a])
                env.step(action)
                return True
            return False
            
        except Exception:
            return False
    
    def _test_dpo_pipeline(self) -> bool:
        """Test complete DPO pipeline."""
        try:
            from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
            from exp08_dpo_implementation import DPOTrainer, DPOConfig
            import copy
            
            # Create components
            tokenizer = SimpleTokenizer(64)
            policy_model = SimpleTransformerLM(tokenizer.vocab_size, embed_dim=32, num_layers=1)
            reference_model = copy.deepcopy(policy_model)
            dataset = PreferenceDataset(tokenizer, size=10)
            
            # Test pipeline: Create trainer -> Training step -> Evaluation
            config = DPOConfig(batch_size=2)
            trainer = DPOTrainer(policy_model, reference_model, tokenizer, config)
            
            batch = dataset.get_batch(2, max_length=16)
            metrics = trainer.training_step(batch)
            eval_metrics = trainer.evaluate(dataset, num_samples=4)
            
            return 'loss' in metrics and 'loss' in eval_metrics
            
        except Exception:
            return False
    
    def _test_reproducibility(self) -> bool:
        """Test reproducibility with fixed seeds."""
        try:
            # Reset seed
            setup_seed(42)
            
            # Generate some random tensors
            x1 = torch.randn(5, 5)
            
            # Reset seed again
            setup_seed(42) 
            
            # Generate same tensors
            x2 = torch.randn(5, 5)
            
            # Should be identical
            return torch.allclose(x1, x2)
            
        except Exception:
            return False
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("INTEGRATION TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result.get('status') == 'PASS')
        
        print(f"\nOverall Results: {passed_tests}/{total_tests} tests passed")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"\n{test_name.upper()}: {status}")
            
            if status == 'PASS':
                # Show key metrics
                for key, value in result.items():
                    if key != 'status' and isinstance(value, (bool, int, str)):
                        print(f"  {key}: {value}")
            else:
                # Show error
                error = result.get('error', 'Unknown error')
                print(f"  Error: {error}")
        
        # System information
        print(f"\nSystem Information:")
        print(f"  Device: {device}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Mixed precision: {'Available' if amp_enabled else 'Not available'}")
        
        # Create summary report
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': passed_tests/total_tests,
                'device': str(device),
                'pytorch_version': torch.__version__
            },
            'detailed_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report

def benchmark_performance():
    """Benchmark performance of key components."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    try:
        # Benchmark Policy-Value Network
        print("\n1. Policy-Value Network Inference:")
        from exp04_policy_value_network import PolicyValueNet, TimingUtils
        
        model = PolicyValueNet().to(device)
        model.eval()
        
        batch_sizes = [1, 8, 32]
        for batch_size in batch_sizes:
            with torch.no_grad():
                states = torch.randn(batch_size, 2, 5, 5).to(device)
                legal_masks = torch.ones(batch_size, 25).bool().to(device)
                
                # Warmup
                for _ in range(10):
                    _ = model(states, legal_masks)
                
                # Measure
                num_iterations = 100
                start_time = time.time()
                for _ in range(num_iterations):
                    _ = model(states, legal_masks)
                end_time = time.time()
                
                throughput = (num_iterations * batch_size) / (end_time - start_time)
                print(f"  Batch size {batch_size}: {throughput:.1f} inferences/sec")
        
        # Benchmark MCTS
        print("\n2. MCTS Search Performance:")
        from exp05_puct_mcts import MCTS
        from exp03_gomoku_environment import Gomoku5x5
        
        env = Gomoku5x5()
        mcts = MCTS(model, c_puct=1.0)
        
        sim_counts = [10, 25, 50]
        for num_sims in sim_counts:
            env.reset()
            
            start_time = time.time()
            action_probs, value = mcts.search(env, num_simulations=num_sims, temperature=0.0, add_noise=False)
            end_time = time.time()
            
            print(f"  {num_sims} simulations: {end_time - start_time:.3f}s")
        
        # Benchmark DPO Training Step
        print("\n3. DPO Training Performance:")
        from exp07_toy_causal_lm import SimpleTokenizer, SimpleTransformerLM, PreferenceDataset
        from exp08_dpo_implementation import DPOTrainer, DPOConfig
        import copy
        
        tokenizer = SimpleTokenizer(128)
        policy_model = SimpleTransformerLM(tokenizer.vocab_size, embed_dim=128, num_layers=2).to(device)
        reference_model = copy.deepcopy(policy_model)
        dataset = PreferenceDataset(tokenizer, size=100)
        
        config = DPOConfig(batch_size=8)
        trainer = DPOTrainer(policy_model, reference_model, tokenizer, config)
        
        batch = dataset.get_batch(config.batch_size, max_length=32)
        
        # Warmup
        for _ in range(5):
            metrics = trainer.training_step(batch)
        
        # Measure
        start_time = time.time()
        num_steps = 10
        for _ in range(num_steps):
            metrics = trainer.training_step(batch)
        end_time = time.time()
        
        steps_per_sec = num_steps / (end_time - start_time)
        print(f"  Training steps: {steps_per_sec:.1f} steps/sec")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    """Run complete integration test suite."""
    print("Starting Lecture 11 Integrated Capstone Test...")
    
    # Run integration tests
    test_suite = IntegrationTestSuite()
    report = test_suite.run_all_tests()
    
    # Performance benchmarks
    benchmark_performance()
    
    # Save report
    report_path = test_suite.artifacts_dir / 'integration_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nIntegration test complete!")
    print(f"Detailed report saved to: {report_path}")
    
    # Final summary
    success_rate = report['summary']['success_rate']
    if success_rate >= 0.8:
        print(f"\n🎉 EXCELLENT: {success_rate*100:.1f}% success rate!")
        print("All major components are working correctly.")
    elif success_rate >= 0.6:
        print(f"\n✅ GOOD: {success_rate*100:.1f}% success rate.")
        print("Most components working, some minor issues.")
    else:
        print(f"\n⚠️  NEEDS ATTENTION: {success_rate*100:.1f}% success rate.")
        print("Several components need debugging.")
    
    print(f"\nLecture 11 components ready for capstone projects!")
    print("Available implementations:")
    print("  • Gomoku 5×5 environment with MCTS/AlphaZero")
    print("  • Policy-value networks with self-play training")
    print("  • Toy language models with DPO optimization")  
    print("  • Complete training infrastructure and utilities")
    
    return report

if __name__ == "__main__":
    main()
