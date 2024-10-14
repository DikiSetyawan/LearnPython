def count_mean(data): 
    """
    This function takes a list of numbers and returns the mean value of the list.
    
    Parameters
    ----------
    data : list
        A list of numbers.
    
    Returns
    -------
    float
        The mean value of the list.
    
    Examples
    --------
    >>> count_mean([1, 2, 3, 4])
    2.5
    """
    total = sum(data)
    length = len(data)
    return total/length

def count_slope(data1, data2):
    """
    This function takes two lists of numbers and returns the slope of the linear regression line.
    
    Parameters
    ----------
    data1 : list
        A list of numbers.
    data2 : list
        A list of numbers.
    
    Returns
    -------
    float
        The slope of the linear regression line.
    
    Examples
    --------
    >>> count_slope([60, 62, 65, 68, 70], [115, 120, 130, 145, 150])
    2.909090909090909
    """
    a = []
    b = []
    mean_1 = count_mean(data1)
    mean_2 = count_mean(data2)
    for i, j in zip(data1, data2): 
        nominator = (i - (mean_1))*(j - (mean_2))
        a.append(nominator)
    for i in data1:
        denom = (i-(mean_1))**2
        b.append(denom)
    
    nomin = sum(a)
    denomin = sum(b)
    slopes_count = nomin/denomin
    print(f"this is slope value {slopes_count}")
    print(f"this is mean of data 1 {mean_1}")
    print(f"this is mean of data 2 {mean_2}")
    return slopes_count
    
def count_intercept(data1, data2):
    """
    Calculate intercept value of linear regression line.
    
    Parameters
    ----------
    data1 : list
        A list of numbers.
    data2 : list
        A list of numbers.
    
    Returns
    -------
    float
        The intercept of the linear regression line.
    
    Examples
    --------
    >>> count_intercept([60, 62, 65, 68, 70], [115, 120, 130, 145, 150])
    54.54545454545455
    """
    return (count_mean(data2) - ((count_slope(data1, data2))*count_mean(data1)))

import matplotlib.pyplot as plt
def linear_regression(data1, data2, input_value): 
    """
    Calculate a predicted value given an input using a linear regression model.
    
    Parameters
    ----------
    data1 : list
        A list of numbers.
    data2 : list
        A list of numbers.
    input_value : float
        The input number to predict.
    
    Returns
    -------
    float
        The predicted value.
    
    Examples
    --------
    >>> linear_regression([60, 62, 65, 68, 70], [115, 120, 130, 145, 150], 66)
    129.54545454545455
    >>> linear_regression([115, 120, 130, 145, 150], [60, 62, 65, 68, 70], 135)
    66.90909090909091
    """
    slope_value = count_slope(data1, data2)
    intercept_val = count_intercept(data1, data2)
    res_linear_regression = intercept_val + (slope_value*(input_value))
    x_values = [min(data1), max(data1)]  # x-axis range
    y_values = [intercept_val + slope_value*x for x in x_values]
    plt.plot(x_values, y_values, 'r-')  # red line
    
    # Plot the data points
    plt.scatter(data1, data2)
    
    plt.xlabel('Height (X)')
    plt.ylabel('Weight (Y)')
    plt.title('Linear Regression Line')
    plt.show()
    return res_linear_regression






x = [60, 62, 65, 68, 70]
y = [115, 120, 130, 145, 150]
predict = linear_regression(x,y, 66)
prdict_2 = linear_regression(y,x, 135)
print(predict)
print(prdict_2)
        



