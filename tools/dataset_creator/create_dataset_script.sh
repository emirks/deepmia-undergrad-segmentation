for i in {1..7}
do
    python create_dataset.py town-${i} --town-index ${i}
done