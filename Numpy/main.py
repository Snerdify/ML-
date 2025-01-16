import numpy as np
# tasks 
'''
BASICS
1. Write an numpy aray and print it and print its type - visualize how numpy arrays are 
just seperated by spaces and are not in a list format

2. Print elements of specific indexes of the arr - same as python list

3. Change the val of an element at certain index and print the array

ATTRIBUTES
4. Write a multi dimensional numpy array and print elements of specific indexes

5. Print the shape of the array

6. Write an arr of shape (2,3,4)

7. print dimensions of the array - how deep the array is . - read more on this 

8. Print the size of the array - total number of elements in the array

9. Print data type in the array , since numpy is written in c , its much faster than python list

DATA TYPES 
10. Write an array with on of the elements as a str - U11 - check data type of an int in that array
- Why does "5" work after typecasting and why does "hello" not work

11. Typecast an array

12. Write a dict , use that dict as an element in array and check the dtype of that arr

13. Have a 3d arr of ints and typecast it to str (<U2)


FILLING ARRAYS:
14. Create an numpy array filled with any value given a certain shape
14.1 filled with zeroes and ones

15. Use empty to reserve space of any shape

16. Use arange to create an array with a range of values

17. use linspace to create an array with a range of values

Nan and Inf- Not a number and infinity
18. use np.nan , np.inf and np.isnan and np.isinf to check if a number is nan or inf
19. WRITE sqrt of -1 and check if its nan , write 1/0 and check if its inf

MATHEMATICAL OPERATIONS
20. multiply python list and numpy arr by 5
21. add a number to a numpy arr , add two numpy arrs 
22. perform subtraction 
( use np.array(list) to convert a list to numpy array)

23. add arr of diff shapes - (1,3) and (2,1)
( why can't we add (1,3) and (2,2))

24. use sqrt , sin , cosin , tan , arctan , log10 on an array  


ARRAY FUNCTIONS 
'''

a = np.array([4,6,7])
a = np.append(a,[9,10,44]) 


a= np.array([[9,8,10],
             [4,5,6]])
# print(a)
# insert [9,7,8] at the 1st index 
c= np.insert(a , 1, [9,7,8])

# delete the element at the 2nd index
# print(np.delete(a, 2))
# print("/n")
# # delete the elements from the 1st row - 0 here specifies a row
# print(np.delete(a , 1 , 0))
# print( "--")
# # delete elements from the 1st column - 1 here specifies a col 
# print(np.delete(a,0,1))



'''
STRUCTURING METHODS

'''
k = np.array([[1,3,4,5,9],
              [8,9,6,7,9],
              [8,5,0,3,4],
              [6,7,8,9,0]])

print(k.shape)
# reshape the above arr in ways compatible with the shape 
# print(k.reshape(5,4))
# print("--")
# print(k.reshape(2,10))
# print("--")
# print(k.reshape(20,1))
# print("--")
# print(k.reshape(20,))
# print("--")
# print(k.reshape(1,1,1,2,10,1,1))


# also use the resize func 


# use the flatten func - returns a flattened copy
# print(k.flatten())

# use ravel - ravel returns a flattened view of the arra
# 
# since flatten just returns a copy , the original arr k is not changed
# var1  = k.flatten()
# var1[2] = 100
# print(var1)
# print(k)


# ravel changes the original arr , here 2nd element in k is permanently changed to 100
# var1  = k.ravel()
# var1[2] = 100
# print(var1)
# print(k)


# use flat to flatten the arr

# var = [ v for v in k.flat]
# print(var) 

# transpose the arr k

# print(k.T)

# use the swapaxes func to swap two axis - this works when the array has lesser dimensions 
# print(k.swapaxes(0,1))


'''
JOINING , STACKING AND SPLITTING ARRAYS
'''
j1 = np.array([[3,5,6] , [9,6,8]])
j2 = np.array([[8,7,7] , [9,8,7]])

# join the two arrays based on axis
j = np.concatenate((j1,j2),axis = 0)
# print(j)
j3 = np.concatenate((j1,j2),axis = 1)
# print(j3)

# USE STACK TO ADD A NEW DIMENSION TO THE ARRs
# print(np.stack((j1,j2)))

# other special types of stacks - hstacks , vstacks 
# k1 = np.array([[1,3,4,5,9,8],
#               [10,9,6,7,9,6],
#               [8,5,0,3,4,7],
#               [6,7,8,9,0,6]])

# use split() to split k1 into 2 arrs ,  rows wise
# print(np.split(k1 , 4 , axis  = 0)) 



'''
AGGREGATE FUNCTIONS 
'''
# min , mean , max , sum , std , mean ,   
# print(k1.mean())
# print(k1.min())
# print(k1.max())
# print(k1.std())
# print(np.median(k1))


'''
RANDOM 
'''
# use random.randint to print a random number
# print(np.random.randint(90))
# use random.randit to print a arr of given shape within the number specified
# print(np.random.randint(100 ,  size = (2,3,4)))
# print("--")
# print between a min and a mx number 
# print(np.random.randint(90,100, size=(2,3,4)))
# print("--")

# print 1s and 0s 
# print(np.random.randint(2, size =(2,3,4)))

# write a binomial distribution 
# print(np.random.binomial(10 , p=0.4 , size = (5,10)))

# use random.normal
# print(np.random.normal(loc=170 , scale=10 , size =(5,10)))

# print(np.random.choice([50,70,67,89],size =(3,4)))

'''
IMPORTING AND EXPORTING NUMPY ARRAYS
'''

# SAVING AN NP ARRAY
# np.save("myarray.npy", k1)

# LOADING THAT NP ARRAY 
k1 = np.load("myarray.npy")
print(k1)   # -> Why is this giving error ? k1 should be loaded from np.load 

# SAVING AN NP ARRAY TO THE CSV FILE 
# np.savetxt("myarray.csv",k1,delimiter=",")
# k1 = np.loadtxt("myarray.csv" , delimiter = ",")
# print(k1)






