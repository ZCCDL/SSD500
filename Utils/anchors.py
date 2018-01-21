
def getscales(m):
    scales = []
    for k in range(1, m+1):
        scales.append(0.2 + (((.7) / (m - 1)) * (k - 1)))
    return scales


def generate_anchors(m):
    return
