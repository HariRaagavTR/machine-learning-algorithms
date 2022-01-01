import numpy as np
import pandas as pd

# input: tuple (x,y) x, y: int
# return a numpy array with one at all index
def create_numpy_ones_array(shape):
    if shape[0] < 0 or shape[1] < 0:
        return None
    return np.ones(shape)

# input: tuple (x,y) x, y: int
# return a numpy array with zeros at all index
def create_numpy_zeros_array(shape):
    if shape[0] < 0 or shape[1] < 0:
        return None
    return np.zeros(shape)

# input: int
# return a identity numpy array of the defined order
def create_identity_numpy_array(order):
    if order < 0:
        return None
    return np.identity(order)

# input: numpy array
# return cofactor matrix of the given array
def matrix_cofactor(array):
    cofactor_matrix = np.zeros(array.shape)
    nrows = array.shape[0]
    ncols = array.shape[1]
    minor_matrix = np.zeros((nrows - 1, ncols - 1))
    for n_row in range(nrows):
        for n_col in range(ncols):
            minor_matrix[:n_row, :n_col] = array[:n_row, :n_col]
            minor_matrix[n_row:, :n_col] = array[n_row + 1:, :n_col]
            minor_matrix[:n_row, n_col:] = array[:n_row, n_col + 1:]
            minor_matrix[n_row:, n_col:] = array[n_row + 1:, n_col + 1:]
            cofactor_matrix[n_row, n_col] = np.linalg.det(minor_matrix)
    return cofactor_matrix
    
        
# Input: (numpy array, int, numpy array, int, int, int, int, tuple, tuple)
# tuple (x, y) x, y: int
# return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) + B
# W1 is random matrix of shape shape1 with seed1
# W2 is random matrix of shape shape2 with seed2
# B is a random matrix of comaptible shape with seed3
def f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    if shape1[1] != X1.shape[0] or shape2[1] != X2.shape[0]:
        return -1
    if shape1[0] != shape2[0] or X1.shape[1] != X2.shape[1]:
        return -1
    
    np.random.seed(seed1)
    W1 = np.random.random_sample(shape1)

    np.random.seed(seed2)
    W2 = np.random.random_sample(shape2)

    np.random.seed(seed3)
    B = np.random.rand(shape1[0], X1.shape[1])

    term1 = np.matmul(W1, X1 ** coef1)
    term2 = np.matmul(W2, X2 ** coef2)
    return term1 + term2 + B

"""
Fill the missing values(NaN) in a column with the mode of that column
Args:
    filename: Name of the CSV file.
    column: Name of the column to fill
Returns:
    df: Pandas DataFrame object.
    (Representing entire data and where 'column' does not contain NaN values)
    (Filled with above mentioned rules)
"""
def fill_with_mode(filename, column):
    try:
        df = pd.read_csv(filename)
        df[column].fillna(value = df[column].mode()[0], inplace = True)
        return df
    except:
        # Returns None if file doesn't exist
        return None

"""
Fill the missing values(NaN) in column with the mean value of the 
group the row belongs to.
The rows are grouped based on the values of another column

Args:
    df: A pandas DataFrame object representing the data.
    group: The column to group the rows with
    column: Name of the column to fill
Returns:
    df: Pandas DataFrame object.
    (Representing entire data and where 'column' does not contain NaN values)
    (Filled with above mentioned rules)
"""
def fill_with_group_average(df, group, column):
    df[column] = df.groupby([group])[column].transform(lambda group: group.fillna(group.mean()))
    return df

"""
Return all the rows (with all columns) where the value in a certain 'column'
is greater than the average value of that column.

row where row.column > mean(data.column)

Args:
    df: A pandas DataFrame object representing the data.
    column: Name of the column to fill
Returns:
    df: Pandas DataFrame object.
"""
def get_rows_greater_than_avg(df, column):
    col_mean = df[column].mean()
    df = df.loc[df[column] > col_mean]
    return df

