# Re-record with: vhs demo/chat.tape   (makes real API calls; needs a built index)
Output demo/chat.gif

Set FontSize 16
Set Width 1200
Set Height 700
Set TypingSpeed 40ms
Set WaitTimeout 60s

Type "uv run cleantech-rag chat"
Enter
Sleep 4s

Type "What are the main challenges for grid-scale battery storage?"
Enter
Wait /You>/

Type "Which of those challenges is the most expensive to solve?"
Enter
Wait /You>/

Type "exit"
Enter
Sleep 1s
