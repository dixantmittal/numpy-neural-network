def getAccuracy(hypothesis, expected):
    totalErrors = 0

    for i in range(0, len(hypothesis)):

        if hypothesis[i] > 0.5:
            output = 1
        else:
            output = 0

        if expected[i] != output:
            totalErrors += 1

    return (len(hypothesis) - totalErrors) / len(hypothesis) * 100
