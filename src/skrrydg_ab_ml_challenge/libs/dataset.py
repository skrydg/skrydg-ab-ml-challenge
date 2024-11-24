
class Dataset:
    def __init__(self, tables):
        self.tables = tables
    
    def get_table(self, name):
        return self.tables[name]
    
    def set_table(self, name, table):
        self.tables[name] = table
