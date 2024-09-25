import pandas as  pd 

# df = pd.read_csv('/home/sat/diki/learnPython/data/daftar-nama-daerah.csv')
# print(df.head())
# df = df[33:-1]
# print(df)
# save = df.to_csv('/home/sat/diki/learnPython/data/daftar-nama-daerah_kabkot.csv', index=False)
df = pd.read_csv('/home/sat/diki/learnPython/data/daftar-nama-daerah_kabkot.csv')
df = df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)

# Optionally, save the modified DataFrame back to a CSV file
df.to_csv('/home/sat/diki/learnPython/data/daftar-nama-daerah_kabkot.csv', index=False)
