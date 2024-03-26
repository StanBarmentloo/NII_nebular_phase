import numpy as np

def main(opened_file):

    header_list = []
    #============================================================================================
    #First we need to get the header from the file
    raw_header = opened_file.read(25000)
    header_start, header_end = 0, 0
    for i in range(25000): #25000 characters should include the full header
        if raw_header[i:i+4] == 'grid':
            header_start = i
        elif raw_header[i:i+5].isnumeric() == True and header_start != 0: #The end of the header is a long empty string
            header_end = i-10
            break
     
    header_string = raw_header[header_start:header_end]
    #============================================================================================
    
    #Now that we have the full header, we get the columns
    column_name_start = 0
    reading_name = True
    for i in range(len(header_string)):
        
        if header_string[i] == ' ' and reading_name == True:
            column_name = header_string[column_name_start:i]
            if column_name != 'stability' and column_name != 'network':
                header_list.append(column_name)
            
            reading_name = False
        
        elif header_string[i] != ' ' and reading_name == False:
            column_name_start = i
            reading_name = True
            
        else:
            pass
    #============================================================================================
        
    return np.array(header_list)
       