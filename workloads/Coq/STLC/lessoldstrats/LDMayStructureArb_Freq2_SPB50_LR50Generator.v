From QuickChick Require Import QuickChick. Import QcNotation.
From Coq Require Import Bool ZArith List. Import ListNotations.
From ExtLib Require Import Monad.
From ExtLib.Data.Monads Require Import OptionMonad.
Import MonadNotation.

From STLC Require Import Impl Spec.

Fixpoint genTyp (size : nat) (stack1 : nat) (stack2 : nat) : G (Typ) :=
  match size with
  | O  => 
    (* Frequency1 (single-branch) *) 
    (returnGen (TBool ))
  | S size1 => 
    (* Frequency2 *) (freq [
      (* TBool *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (returnGen (TBool ))); 
      (* TFun *) (match (size, stack1, stack2) with
      | (1, 2, 1) => 50
      | (1, 2, 4) => 50
      | (2, 0, 2) => 50
      | (2, 3, 2) => 50
      | (2, 5, 2) => 50
      | (2, 6, 2) => 50
      | _ => 500
      end,
      (bindGen (genTyp size1 stack2 1) 
      (fun p1 => 
        (bindGen (genTyp size1 stack2 4) 
        (fun p2 => 
          (returnGen (TFun p1 p2)))))))])
  end.

Fixpoint genExpr (size : nat) (stack1 : nat) (stack2 : nat) : G (Expr) :=
  match size with
  | O  => 
    (* Frequency3 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 91
      | (0, 3, 5) => 76
      | (0, 3, 6) => 95
      | (0, 5, 3) => 89
      | (0, 5, 5) => 89
      | (0, 5, 6) => 86
      | (0, 6, 3) => 90
      | (0, 6, 5) => 96
      | (0, 6, 6) => 96
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 98
      | (0, 3, 6) => 95
      | (0, 5, 3) => 0
      | (0, 5, 5) => 45
      | (0, 5, 6) => 84
      | (0, 6, 3) => 0
      | (0, 6, 5) => 93
      | (0, 6, 6) => 90
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 43
      | (1, 3, 5) => 51
      | (1, 3, 6) => 57
      | (1, 5, 3) => 53
      | (1, 5, 5) => 11
      | (1, 5, 6) => 17
      | (1, 6, 3) => 80
      | (1, 6, 5) => 73
      | (1, 6, 6) => 37
      | (2, 3, 3) => 2
      | (2, 3, 5) => 1
      | (2, 3, 6) => 1
      | (2, 5, 3) => 53
      | (2, 5, 5) => 44
      | (2, 5, 6) => 54
      | (2, 6, 3) => 2
      | (2, 6, 5) => 10
      | (2, 6, 6) => 3
      | (3, 3, 3) => 30
      | (3, 3, 5) => 50
      | (3, 3, 6) => 31
      | (3, 5, 3) => 0
      | (3, 5, 5) => 26
      | (3, 5, 6) => 0
      | (3, 6, 3) => 43
      | (3, 6, 5) => 41
      | (3, 6, 6) => 43
      | (4, 0, 3) => 7
      | (4, 0, 5) => 0
      | (4, 0, 6) => 8
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 38
      | (1, 3, 6) => 1
      | (1, 5, 3) => 0
      | (1, 5, 5) => 67
      | (1, 5, 6) => 5
      | (1, 6, 3) => 0
      | (1, 6, 5) => 58
      | (1, 6, 6) => 6
      | (2, 3, 3) => 0
      | (2, 3, 5) => 5
      | (2, 3, 6) => 1
      | (2, 5, 3) => 11
      | (2, 5, 5) => 47
      | (2, 5, 6) => 46
      | (2, 6, 3) => 0
      | (2, 6, 5) => 2
      | (2, 6, 6) => 0
      | (3, 3, 3) => 30
      | (3, 3, 5) => 50
      | (3, 3, 6) => 53
      | (3, 5, 3) => 0
      | (3, 5, 5) => 26
      | (3, 5, 6) => 0
      | (3, 6, 3) => 43
      | (3, 6, 5) => 48
      | (3, 6, 6) => 60
      | (4, 0, 3) => 7
      | (4, 0, 5) => 0
      | (4, 0, 6) => 68
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 96
      | (1, 3, 5) => 90
      | (1, 3, 6) => 98
      | (1, 5, 3) => 93
      | (1, 5, 5) => 76
      | (1, 5, 6) => 87
      | (1, 6, 3) => 98
      | (1, 6, 5) => 1
      | (1, 6, 6) => 99
      | (2, 3, 3) => 10
      | (2, 3, 5) => 30
      | (2, 3, 6) => 1
      | (2, 5, 3) => 72
      | (2, 5, 5) => 63
      | (2, 5, 6) => 50
      | (2, 6, 3) => 15
      | (2, 6, 5) => 2
      | (2, 6, 6) => 44
      | (3, 3, 3) => 79
      | (3, 3, 5) => 45
      | (3, 3, 6) => 69
      | (3, 5, 3) => 0
      | (3, 5, 5) => 62
      | (3, 5, 6) => 0
      | (3, 6, 3) => 64
      | (3, 6, 5) => 60
      | (3, 6, 6) => 52
      | (4, 0, 3) => 87
      | (4, 0, 5) => 0
      | (4, 0, 6) => 79
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 98
      | (1, 3, 5) => 94
      | (1, 3, 6) => 95
      | (1, 5, 3) => 5
      | (1, 5, 5) => 51
      | (1, 5, 6) => 78
      | (1, 6, 3) => 90
      | (1, 6, 5) => 95
      | (1, 6, 6) => 95
      | (2, 3, 3) => 97
      | (2, 3, 5) => 92
      | (2, 3, 6) => 97
      | (2, 5, 3) => 57
      | (2, 5, 5) => 45
      | (2, 5, 6) => 53
      | (2, 6, 3) => 97
      | (2, 6, 5) => 93
      | (2, 6, 6) => 96
      | (3, 3, 3) => 30
      | (3, 3, 5) => 57
      | (3, 3, 6) => 39
      | (3, 5, 3) => 96
      | (3, 5, 5) => 72
      | (3, 5, 6) => 96
      | (3, 6, 3) => 47
      | (3, 6, 5) => 52
      | (3, 6, 6) => 43
      | (4, 0, 3) => 42
      | (4, 0, 5) => 96
      | (4, 0, 6) => 18
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen (genExpr size1 stack2 3) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 6) 
        (fun p2 => 
          (returnGen (App p1 p2)))))))])
  end.

Definition gSized :=
  (genExpr 5 0 0).

Definition test_prop_SinglePreserve :=
forAll gSized (fun (e: Expr) =>
  prop_SinglePreserve e).

(*! QuickChick test_prop_SinglePreserve. *)

Definition test_prop_MultiPreserve :=
forAll gSized (fun (e: Expr) =>
  prop_MultiPreserve e).

(*! QuickChick test_prop_MultiPreserve. *)
          
