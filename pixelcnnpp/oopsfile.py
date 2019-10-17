fin=open('results.csv','r')
fout=open('resultsEdited.csv','w')
    
for line in fin:
    if line[0] != '0':
        fout.write(line+"\n")
    else:
        i=1
        arr=line.split(",")
        count=0
        currentLine=""
        for elem in arr:
            if count==4:
                print(currentLine)
                if int(elem[-1])==i:
                    currentLine+=elem[:-1]

                    fout.write(currentLine+"\n")
                    currentLine=elem[-1]+","
                elif  int(elem[-2:])==i:
                    currentLine+=elem[:-2] 
                    fout.write(currentLine+"\n")
                    
                    currentLine=elem[-2:]+","
                elif int(elem[-3:])==i:
                    currentLine+=elem[:-3]
                    fout.write(currentLine+"\n")
                    currentLine=elem[-3:]+","
                    
                count=1
                i+=1
            else:
                currentLine+=elem+","
                count+=1         
fin.close()
fout.close()                         
