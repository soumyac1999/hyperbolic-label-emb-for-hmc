for i in `ls -v $1`; do 
	echo -n $i \ ;
	tail -n 1 $1/$i/res.txt | tr -d [:alpha:] | tr -d \'\(\)\:\{\}\ | cut -d ',' -f 4,5; 
done
