if [ "$1" == "train" ]; then
  txt='train.txt'
  path='./train_vid'
elif [ "$1" == "val" ]; then
  txt='val.txt'
  path='./val_vid'
fi

mkdir $path
while read line;
do
  wget "$line"
  tar zxvf "${line##*/}"
  rm "${line##*/}"
  mv ./*.mp4 $path
done < $txt
