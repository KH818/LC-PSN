"""Prompt for a yes-or-no choice in legacy baseline training code."""

def ask(question):
    answer = input(f"{question} (Y/n): ").strip().lower()
    if answer in ['y', 'yes', '']:
        return True
    else:
        return False
