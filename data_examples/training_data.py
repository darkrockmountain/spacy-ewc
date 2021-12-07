new_labels=["BUDDY", "COMPANY"]

# Set the training data for all integration test for consistency:
training_data = [
    ("John Doe is a person.", {"entities": [(0, 8, "BUDDY")]}),
    ("OpenAI is a company.", {"entities": [(0, 6, "COMPANY")]}),
    ("Jane Smith works at Microsoft.", {
        "entities": [(0, 10, "BUDDY"), (20, 29, "COMPANY")]}),
    ("GMaps was developed at Google.", {"entities": [(23, 29, "COMPANY")]}),
    ("John Doe works at OpenAI.", {
        "entities": [(0, 8, "BUDDY"), (18, 24, "COMPANY")]}),
    ("John Doe and Jane Smith are colleagues at OpenAI.", {
        "entities": [(0, 8, "BUDDY"), (13, 23, "BUDDY"), (42, 48, "COMPANY")]}),
    ("OpenAI hired John Doe.", {
        "entities": [(0, 6, "COMPANY"), (13, 21, "BUDDY")]}),
    # Corrected  entries
    ("Alice Johnson joined Amazon last year.", {
        "entities": [(0, 13, "BUDDY"), (21, 27, "COMPANY")]}),
    ("Bob Lee and Carol King work for Facebook.", {
        "entities": [(0, 7, "BUDDY"), (12, 22, "BUDDY"), (32, 40, "COMPANY")]}),
    ("Diana Prince was promoted at DC Comics.", {
        "entities": [(0, 12, "BUDDY"), (29, 38, "COMPANY")]}),
    ("Evan Wright is the CEO of Tesla.", {
        "entities": [(0, 11, "BUDDY"), (26, 31, "COMPANY")]}),
    ("Frank Miller collaborates with Pixar Studios.", {
        "entities": [(0, 12, "BUDDY"), (31, 44, "COMPANY")]}),
    ("Grace Hopper founded Grace Computing.", {
        "entities": [(0, 12, "BUDDY"), (21, 36, "COMPANY")]}),
    ("Henry Ford started Ford Motor Company.", {
        "entities": [(0, 10, "BUDDY"), (19, 37, "COMPANY")]}),
    ("Ivy Chen joined Apple Inc. in 2020.", {
        "entities": [(0, 8, "BUDDY"), (16, 26, "COMPANY")]}),
    ("Jack Black works at Netflix.", {
        "entities": [(0, 10, "BUDDY"), (20, 27, "COMPANY")]}),
    ("Karen Gillan is a star at Universal Pictures.", {
        "entities": [(0, 12, "BUDDY"), (26, 44, "COMPANY")]}),
    ("Leo Messi plays for Paris Saint-Germain.", {
        "entities": [(0, 9, "BUDDY"), (20, 39, "COMPANY")]}),
    ("Maria Garcia leads the team at IBM.", {
        "entities": [(0, 12, "BUDDY"), (31, 34, "COMPANY")]}),
    ("Nathan Drake works with Ubisoft.", {
        "entities": [(0, 12, "BUDDY"), (24, 32, "COMPANY")]}),
    ("Olivia Brown was hired by Starbucks.", {
        "entities": [(0, 12, "BUDDY"), (26, 35, "COMPANY")]}),
    ("Paul Allen co-founded Microsoft.", {
        "entities": [(0, 10, "BUDDY"), (22, 31, "COMPANY")]}),
    ("Quincy Adams is employed at Amazon.", {
        "entities": [(0, 12, "BUDDY"), (28, 34, "COMPANY")]}),
    ("Rachel Green works for Nike.", {
        "entities": [(0, 12, "BUDDY"), (23, 27, "COMPANY")]}),
    ("Steve Rogers is associated with Stark Industries.", {
        "entities": [(0, 12, "BUDDY"), (32, 48, "COMPANY")]}),
    ("Tom Hanks stars in movies by Paramount Pictures.", {
        "entities": [(0, 9, "BUDDY"), (29, 48, "COMPANY")]}),
    ("Uma Thurman is a leading actress at Disney.", {
        "entities": [(0, 11, "BUDDY"), (36, 42, "COMPANY")]}),
    ("Victor Stone works at Cyberdyne Systems.", {
        "entities": [(0, 12, "BUDDY"), (22, 39, "COMPANY")]}),
    ("Wendy Wu joined Tencent.", {
        "entities": [(0, 8, "BUDDY"), (16, 23, "COMPANY")]}),
    ("Xavier Woods is part of WWE.", {
        "entities": [(0, 12, "BUDDY"), (24, 27, "COMPANY")]}),
    ("Yvonne Strahovski acts for Paramount.", {
        "entities": [(0, 17, "BUDDY"), (27, 36, "COMPANY")]}),
    ("Zachary Quinto works at HBO.", {
        "entities": [(0, 14, "BUDDY"), (24, 27, "COMPANY")]}),
    # Add more variations as needed
]