if [ -z $1 ]
then
  dir=logs
else
  dir=$1
fi

if [ -z $2 ]
then
  ver=0.8.0
else
  ver=$2
fi

mkdir -p "$dir"

wget -r -np -nH --cut-dirs=2 -P "$dir" -R "index.html*" https://martin-krasser.com/perceiver/logs-$ver/
