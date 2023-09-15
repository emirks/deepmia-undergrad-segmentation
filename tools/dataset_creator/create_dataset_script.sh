for i in {1..7}
do
    python create_dataset.py town-${i} --town-index ${i} --number-of-egos 195 --frames-per-ego 1
    python create_dataset.py town-${i}-val --town-index ${i} --number-of-egos 8 --frames-per-ego 3
    python create_dataset.py town-${i}-test --town-index ${i} --number-of-egos 8 --frames-per-ego 3
done