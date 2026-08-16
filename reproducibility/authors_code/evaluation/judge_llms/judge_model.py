
class JudgeModel:
    '''
    Base class for all judge model adapters
    '''
    def __init__(self,  model_id, device='cuda'):
        self.device = device
        self.model_id = model_id
        self.model, self.tokenizer = self.load(model_id)
        
    def load(self, model_id):
        '''
        loads model and tokenizer and whatever else is necessary,
        returns model and tokenizer
        '''
        raise NotImplementedError

    def evaluate(self, query, response):
        '''
        evaluate the query-response pair, returning the raw model output
        '''
        raise NotImplementedError
    
    def parse_evaluation(self, evaluation):
        '''
        parse the model output into boolean judgement:
        True - unsafe response, attack success
        False - safe response, attack fail
        '''
        raise NotImplementedError
    
