#!/bin/bash

# Script to retry the PINGS command until it succeeds
# Success is defined as NOT ending with "Lose track for a long time, system failed"

COMMAND="python pings.py ./config/run_agri_slam_gs.yaml agri_slam -i 2d-apple_harvesting_train_c10_l20 -s"
ATTEMPT=1
MAX_ATTEMPTS=50  # Set a reasonable limit to prevent infinite loops

echo "Starting retry script for PINGS command..."
echo "Command: $COMMAND"
echo "Will retry until output doesn't end with 'Lose track for a long time, system failed'"
echo "Maximum attempts: $MAX_ATTEMPTS"
echo "----------------------------------------"

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS at $(date)"
    echo "Running command..."
    
    # Run the command and capture both stdout and stderr
    OUTPUT=$(eval "$COMMAND" 2>&1)
    EXIT_CODE=$?
    
    echo "Command finished with exit code: $EXIT_CODE"
    
    # Check if the output ends with the failure message
    if echo "$OUTPUT" | tail -1 | grep -q "Lose track for a long time, system failed"; then
        echo "❌ Attempt $ATTEMPT failed - output ends with failure message"
        echo "Last few lines of output:"
        echo "$OUTPUT" | tail -5
        echo "----------------------------------------"
        
        # Wait a bit before retrying
        sleep 2
        ATTEMPT=$((ATTEMPT + 1))
    else
        echo "✅ Success! Command completed without the failure message"
        echo "Final output:"
        echo "$OUTPUT"
        echo "----------------------------------------"
        echo "Script completed successfully after $ATTEMPT attempts"
        exit 0
    fi
done

echo "❌ Maximum attempts ($MAX_ATTEMPTS) reached. Script stopping."
echo "Last output was:"
echo "$OUTPUT" | tail -10
exit 1