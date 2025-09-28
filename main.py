import numpy as np




def get_first_nonzero_value_for_row(array,row,augmented=False):
    if augmented==True:
        array=array[:,:-1]
    research_array=array[row]
    for i , value in enumerate(research_array):
        if not np.isclose(value,0,atol=1e-5):
            return i
    return -1


def swap_rows(M, row_index_1, row_index_2):
    M = M.copy()
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M
def get_index_first_non_zero_value_from_column(array,column,starting_row=0):
    research_array=array[starting_row:,column]
    for  i , value in enumerate(research_array):
        if not np.isclose(value,0,atol=1e-5):
            return i+ starting_row
    return -1
def augnented_matrix(array1,array2):
    return np.hstack((array1,array2))
def row_echelon_form(A,B):
    if np.isclose(np.linalg.det(A),0)==True:
        return "Singular System"
    
    A=A.copy()
    B=B.copy()
    A=A.astype('float64')
    B=B.astype('float64')
    
    num_rows=len(A)

    M=augnented_matrix(A,B)

    for row in range(num_rows):
        pivot_candidate=M[row,row]
        if np.isclose(pivot_candidate,0) ==True:
            first_pivot_candidate_in_the_column=get_index_first_non_zero_value_from_column(A,row,row+1)
            M=swap_rows(M,row,first_pivot_candidate_in_the_column)
            pivot=A[row,row]
        else:
            pivot=pivot_candidate
        

        M[row]=M[row]/pivot



        for j in range(row+1,num_rows):
            value_below_pivot=M[j,row]


            M[j]=M[j]-value_below_pivot*M[row]
    
    
    return M







def back_substitution(M):


    M=M.copy()

    num_rows=M.shape[0]

    for row in reversed(range(num_rows)):
        substitution_row=M[row]
        index=get_first_nonzero_value_for_row(M,row,augmented=True)


        for j in range(row):
            row_to_reduce=M[j]
            value=row_to_reduce[index]
            row_to_reduce=row_to_reduce - value*substitution_row
            M[j,:]=row_to_reduce


    solution=M[:,-1]


    return solution
def gaussian_elimination(A, B):
    row_echelon_M=row_echelon_form(A,B)

    solution=back_substitution(row_echelon_M)

    return solution





# variable,A,B=
"""
there must be your variables like a,b,c,w,y and etc. Normal matrix and its constant part
"""

A = np.array([
    [3, 6, 6, 8],
    [5, 3, 6, 0],
    [0, 4, -5, 8],
    [0, 0, 4, 8]
], dtype=float)  
B = np.array([
    [1],
    [-10],
    [8],
    [9]
], dtype=float)
variables = ['x', 'y', 'w', 'z']
sols=gaussian_elimination(A,B)
if not isinstance(sols,str):
    for variable, val in enumerate(sols):
        print(f'{variable} = {val}')
else:
    print(sols)