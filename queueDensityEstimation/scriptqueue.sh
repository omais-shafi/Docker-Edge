COUNTER=1
while [ $COUNTER -le 50 ]; do
             ./code_no_dyn.o Approach3_Back_2019_11_23_03_58_08-Small-Day.mp4 0 0 0 0 >> CPUvaryresults/$1
             COUNTER=$((COUNTER+1))
done






