"""
Shared utilities for the MECH framework.
"""


class EarlyStopping:
    """Early stopping to terminate training when validation metric stops improving."""

    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
                return True
            else:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
                return False


TEACHER_NAMES = {'T', 'Teacher', 'Ms. G', 'Mrs. G'}


def get_role_name(speaker_name, current_speaker_name, teacher_names=None):
    """Get relative role marker for a speaker."""
    if teacher_names is None:
        teacher_names = TEACHER_NAMES
    s_name = str(speaker_name).strip()
    if s_name.startswith('T') or s_name in teacher_names:
        return "[TEACHER]"
    if s_name == str(current_speaker_name).strip():
        return "[CURRENT]"
    return "[OTHER]"
