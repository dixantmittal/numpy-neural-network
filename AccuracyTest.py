def getAccuracy(hypothesis, expected):
    totalErrors = 0

    for i in range(0, len(hypothesis)):
        if expected[i] != hypothesis[i]:
            totalErrors += 1

    return (len(hypothesis) - totalErrors) / len(hypothesis) * 100
