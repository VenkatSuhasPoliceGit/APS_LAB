import numpy as np
import matplotlib.pyplot as plt


print("\nPROBLEM 1: FAIR COIN TOSSED 100 TIMES")

coin = np.random.randint(0, 2, 100)

heads = np.sum(coin == 1)
tails = np.sum(coin == 0)

p_head = heads / 100
p_tail = tails / 100

print("Number of Heads:", heads)
print("Number of Tails:", tails)
print("Experimental Probability of Head:", p_head)
print("Experimental Probability of Tail:", p_tail)



print("\nPROBLEM 2: EFFECT OF NUMBER OF TRIALS")

trials = [10, 100, 1000, 10000]
head_prob = []

for n in trials:
    coin = np.random.randint(0, 2, n)

    heads = np.sum(coin == 1)
    tails = np.sum(coin == 0)

    p_head = heads / n
    p_tail = tails / n
    head_prob.append(p_head)

    print("\nNumber of Trials:", n)
    print("Heads:", heads)
    print("Tails:", tails)
    print("Probability of Head:", p_head)
    print("Probability of Tail:", p_tail)

plt.figure()
plt.plot(trials, head_prob, marker="o", label="Experimental P(Head)")
plt.axhline(0.5, linestyle="--", label="Theoretical P(Head) = 0.5")
plt.xlabel("Number of Trials")
plt.ylabel("Probability of Head")
plt.title("Effect of Number of Trials")
plt.legend()
plt.grid(True)
plt.show()



print("\nPROBLEM 3: ROLLING A FAIR DIE 6000 TIMES")

die = np.random.randint(1, 7, 6000)
counts = []

for face in range(1, 7):
    count = np.sum(die == face)
    probability = count / 6000
    counts.append(count)

    print("Face:", face)
    print("Count:", count)
    print("Experimental Probability:", probability)

plt.figure()
plt.bar(range(1, 7), counts)
plt.xlabel("Die Face")
plt.ylabel("Frequency")
plt.title("Frequency of Each Die Outcome")
plt.xticks(range(1, 7))
plt.show()



print("\nPROBLEM 4: ADDITION RULE FOR DISJOINT EVENTS")

die = np.random.randint(1, 7, 10000)

A = (die == 1)  # Getting 1
B = (die == 6)  # Getting 6

p_A = np.mean(A)
p_B = np.mean(B)
p_union = np.mean(A | B)

print("Events A and B are disjoint.")
print("P(A):", p_A)
print("P(B):", p_B)
print("P(A union B):", p_union)
print("P(A) + P(B):", p_A + p_B)
print("Verification:", np.isclose(p_union, p_A + p_B))



print("\nPROBLEM 5: ADDITION RULE FOR OVERLAPPING EVENTS")

S = {1, 2, 3, 4, 5, 6}
A = {2, 4, 6}  
B = {4, 5, 6}  

intersection = A & B
union = A | B

p_A = len(A) / len(S)
p_B = len(B) / len(S)
p_intersection = len(intersection) / len(S)
p_union = len(union) / len(S)

print("Sample Space:", S)
print("Event A:", A)
print("Event B:", B)
print("Intersection A and B:", intersection)
print("Union A or B:", union)
print("P(A):", p_A)
print("P(B):", p_B)
print("P(A intersection B):", p_intersection)
print("P(A union B):", p_union)
print("P(A) + P(B) - P(A intersection B):",
      p_A + p_B - p_intersection)
print("Verification:",
      np.isclose(p_union, p_A + p_B - p_intersection))



print("\nPROBLEM 6: MULTIPLICATION RULE FOR INDEPENDENT EVENTS")

n = 10000

coin = np.random.randint(0, 2, n)
die = np.random.randint(1, 7, n)

A = (coin == 1)       # Obtaining Head
B = (die % 2 == 0)    # Obtaining an even number

p_A = np.mean(A)
p_B = np.mean(B)
p_intersection = np.mean(A & B)

print("P(A) - Head:", p_A)
print("P(B) - Even Number:", p_B)
print("P(A intersection B):", p_intersection)
print("P(A) * P(B):", p_A * p_B)
print("Verification (approximately):",
      np.isclose(p_intersection, p_A * p_B, atol=0.02))
