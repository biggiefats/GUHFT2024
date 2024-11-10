import numpy as np
import pandas as pd

class TransactionTracker():
    def __init__(self):
        """
        A basic habit tracker app that uses ML to track expenses!
        """
        self.df = pd.read_csv(r'data.csv')

    def add_entry(self, list):
        """Add an entry to the dataframe."""
        self.df.loc[len(self.df), :] = list
        print(f"Added new entry: {list}")
        self.df.to_csv('data.csv', index=False)

    def remove_entry(self, name):
        """Remove an entry from the list, given a name."""
        i = list(self.df['name'].values).index(name)
        self.df.drop(i)
        print(f"Removed expense: {name} from data.")
        self.df.to_csv('data.csv', index=False)

tt = TransactionTracker()
tt.add_entry(['ICloud', '1', 'M', 'H'])
