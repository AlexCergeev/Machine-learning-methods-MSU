def check(s, filename):
  letters_frequencies = {}
  word = s.lower().split()
  for el in word:
    if el in letters_frequencies:
      letters_frequencies[el] += 1
    else:
      letters_frequencies[el] = 1

  with open(filename, 'w') as file:
      for i, j in sorted(letters_frequencies.items()):
          print(i, j, file=file)
