def init():
    global metrics
    global model
    global f1_score
    global scheduler
    global optimizer

    metrics = []
    model, f1_score = None, 0.
    scheduler, optimizer = None, None
