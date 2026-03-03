"""
Training callbacks and GPU/RAM monitoring helpers.
"""

import time
from typing import Optional

from transformers import TrainerCallback

# ── optional monitoring libs ──────────────────────────────────────────────────
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except Exception:
    GPU_MONITORING_AVAILABLE = False

try:
    import psutil
    RAM_MONITORING_AVAILABLE = True
except ImportError:
    RAM_MONITORING_AVAILABLE = False


# ============================================================================
# Hardware stats helpers
# ============================================================================

def get_gpu_memory_stats() -> dict:
    if not GPU_MONITORING_AVAILABLE:
        return {}
    try:
        handle   = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'gpu_vram_total_gb': mem_info.total / 1e9,
            'gpu_vram_used_gb':  mem_info.used  / 1e9,
            'gpu_vram_free_gb':  mem_info.free  / 1e9,
            'gpu_vram_percent':  (mem_info.used / mem_info.total) * 100,
        }
    except Exception:
        return {}


def get_cpu_ram_stats() -> dict:
    if not RAM_MONITORING_AVAILABLE:
        return {}
    try:
        ram = psutil.virtual_memory()
        return {
            'cpu_ram_total_gb':    ram.total / 1e9,
            'cpu_ram_used_gb':     ram.used  / 1e9,
            'cpu_ram_available_gb': ram.available / 1e9,
            'cpu_ram_percent':     ram.percent,
        }
    except Exception:
        return {}


# ============================================================================
# Callback
# ============================================================================

class DetailedLoggingCallback(TrainerCallback):
    """
    Logs step-level progress to the console (and optionally to WandB) and
    prints epoch / training summaries.
    """

    def __init__(self, log_every_n_steps: int = 50, wandb_enabled: bool = False):
        self.log_every_n_steps   = log_every_n_steps
        self.wandb_enabled       = wandb_enabled
        self.epoch_start_time: Optional[float]    = None
        self.training_start_time: Optional[float] = None
        self._step_times: list = []

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        print('\n' + '='*80)
        print('TRAINING STARTED')
        print('='*80)
        print(f"  Epochs: {args.num_train_epochs} | "
              f"Steps: {state.max_steps} | "
              f"LR: {args.learning_rate} | "
              f"Batch (per device): {args.per_device_train_batch_size} × "
              f"{args.gradient_accumulation_steps} accum")
        print('='*80 + '\n')

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        epoch = int(state.epoch) if state.epoch is not None else 0
        print(f'\n{"="*80}')
        print(f'EPOCH {epoch + 1}/{int(args.num_train_epochs)}')
        print(f'{"="*80}')

    def on_step_end(self, args, state, control, **kwargs):
        self._step_times.append(time.time())
        if state.global_step % self.log_every_n_steps != 0 and state.global_step != 1:
            return

        # Timing / ETA
        if len(self._step_times) > 10:
            avg_step = (self._step_times[-1] - self._step_times[-10]) / 10
        elif len(self._step_times) > 1:
            avg_step = (self._step_times[-1] - self._step_times[0]) / len(self._step_times)
        else:
            avg_step = 0

        remaining_steps = state.max_steps - state.global_step
        eta_h = remaining_steps * avg_step / 3600

        loss = lr = None
        for entry in reversed(state.log_history):
            if loss is None and 'loss' in entry:
                loss = entry['loss']
            if lr is None and 'learning_rate' in entry:
                lr = entry['learning_rate']
            if loss is not None and lr is not None:
                break

        epoch = int(state.epoch) if state.epoch is not None else 0
        line  = f"Step {state.global_step}/{state.max_steps} | Epoch {epoch+1}"
        if loss is not None:  line += f" | Loss: {loss:.4f}"
        if lr   is not None:  line += f" | LR: {lr:.2e}"
        if avg_step > 0:      line += f" | {avg_step:.2f}s/step | ETA: {eta_h:.1f}h"
        print(line)

        gpu = get_gpu_memory_stats()
        ram = get_cpu_ram_stats()
        if gpu or ram:
            parts = []
            if gpu: parts.append(f"GPU: {gpu.get('gpu_vram_used_gb',0):.1f}/"
                                  f"{gpu.get('gpu_vram_total_gb',0):.1f}GB")
            if ram: parts.append(f"RAM: {ram.get('cpu_ram_used_gb',0):.1f}/"
                                  f"{ram.get('cpu_ram_total_gb',0):.1f}GB")
            print("  └─ " + " | ".join(parts))

        if self.wandb_enabled:
            try:
                import wandb
                metrics: dict = {'train/step': state.global_step, 'train/epoch': state.epoch}
                if loss is not None:  metrics['train/loss'] = loss
                if lr   is not None:  metrics['train/learning_rate'] = lr
                if avg_step > 0:      metrics['train/step_time'] = avg_step
                metrics.update({f'train/{k}': v for k, v in gpu.items()})
                metrics.update({f'train/{k}': v for k, v in ram.items()})
                wandb.log(metrics)
            except Exception as e:
                print(f"  WandB error: {e}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        print(f'\n{"─"*80}\nEVALUATION\n{"─"*80}')
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f'{"─"*80}\n')

        if self.wandb_enabled:
            try:
                import wandb
                wandb.log({f'eval/{k.replace("eval_", "")}': v for k, v in metrics.items()})
            except Exception:
                pass

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch = int(state.epoch) if state.epoch is not None else 0
            mins  = (time.time() - self.epoch_start_time) / 60
            print(f'\n{"─"*80}')
            print(f'EPOCH {epoch}/{int(args.num_train_epochs)} done in {mins:.1f} min')
            print(f'{"─"*80}\n')

    def on_train_end(self, args, state, control, **kwargs):
        if self.training_start_time:
            hours = (time.time() - self.training_start_time) / 3600
            avg   = (time.time() - self.training_start_time) / max(1, state.global_step)
            print(f'\n{"="*80}')
            print(f'TRAINING COMPLETE  —  {hours:.2f}h total, {avg:.2f}s/step')
            print(f'{"="*80}\n')
