from app import db

class MathProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    topic = db.Column(db.String(50))
    difficulty = db.Column(db.String(20))
    completed = db.Column(db.Integer, default=0)
    correct = db.Column(db.Integer, default=0)
    last_attempt = db.Column(db.DateTime)

    def __repr__(self):
        return f'<MathProgress {self.topic} {self.difficulty}>'
