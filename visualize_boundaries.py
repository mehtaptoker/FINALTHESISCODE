#####This code is debugged and adapted with help of G00GLE GEMINI############
import matplotlib.pyplot as plt

# Boundaries from Example1_processed.json
boundaries = [[50.0, -21.39689578713969], [-50.0, -0.3325942350332589], [-50.0, 21.39689578713969], [50.0, 21.39689578713969]]
input_shaft = (36.30761873229591, 1.5261878246096288)
output_shaft = (-32.082486095074614, 11.056672150299214)

# Extract x and y coordinates
x = [point[0] for point in boundaries]
y = [point[1] for point in boundaries]
x.append(boundaries[0][0])  # Close the polygon
y.append(boundaries[0][1])

# Create plot
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', label='Boundaries')
plt.plot(input_shaft[0], input_shaft[1], 'go', markersize=10, label='Input Shaft')
plt.plot(output_shaft[0], output_shaft[1], 'ro', markersize=10, label='Output Shaft')
plt.title('Gear Pathfinding Boundaries')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('boundaries_visualization.png')
print("Visualization saved to boundaries_visualization.png")
