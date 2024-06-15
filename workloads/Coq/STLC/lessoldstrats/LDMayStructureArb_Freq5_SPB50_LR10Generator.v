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
      | (0, 3, 3) => 88
      | (0, 3, 5) => 61
      | (0, 3, 6) => 70
      | (0, 5, 3) => 86
      | (0, 5, 5) => 44
      | (0, 5, 6) => 61
      | (0, 6, 3) => 88
      | (0, 6, 5) => 45
      | (0, 6, 6) => 75
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (0, 3, 3) => 0
      | (0, 3, 5) => 62
      | (0, 3, 6) => 44
      | (0, 5, 3) => 0
      | (0, 5, 5) => 61
      | (0, 5, 6) => 45
      | (0, 6, 3) => 0
      | (0, 6, 5) => 69
      | (0, 6, 6) => 22
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1)))))])
  | S size1 => 
    (* Frequency4 *) (freq [
      (* Var *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 17
      | (1, 3, 5) => 42
      | (1, 3, 6) => 50
      | (1, 5, 3) => 23
      | (1, 5, 5) => 33
      | (1, 5, 6) => 59
      | (1, 6, 3) => 70
      | (1, 6, 5) => 41
      | (1, 6, 6) => 74
      | (2, 3, 3) => 3
      | (2, 3, 5) => 17
      | (2, 3, 6) => 2
      | (2, 5, 3) => 53
      | (2, 5, 5) => 31
      | (2, 5, 6) => 57
      | (2, 6, 3) => 4
      | (2, 6, 5) => 10
      | (2, 6, 6) => 1
      | (3, 3, 3) => 37
      | (3, 3, 5) => 44
      | (3, 3, 6) => 37
      | (3, 5, 3) => 0
      | (3, 5, 5) => 23
      | (3, 5, 6) => 0
      | (3, 6, 3) => 45
      | (3, 6, 5) => 41
      | (3, 6, 6) => 45
      | (4, 0, 3) => 9
      | (4, 0, 5) => 0
      | (4, 0, 6) => 10
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Var p1))))); 
      (* Bool *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 0
      | (1, 3, 5) => 61
      | (1, 3, 6) => 46
      | (1, 5, 3) => 1
      | (1, 5, 5) => 42
      | (1, 5, 6) => 17
      | (1, 6, 3) => 0
      | (1, 6, 5) => 10
      | (1, 6, 6) => 22
      | (2, 3, 3) => 0
      | (2, 3, 5) => 47
      | (2, 3, 6) => 1
      | (2, 5, 3) => 7
      | (2, 5, 5) => 31
      | (2, 5, 6) => 38
      | (2, 6, 3) => 0
      | (2, 6, 5) => 11
      | (2, 6, 6) => 2
      | (3, 3, 3) => 37
      | (3, 3, 5) => 45
      | (3, 3, 6) => 59
      | (3, 5, 3) => 0
      | (3, 5, 5) => 14
      | (3, 5, 6) => 0
      | (3, 6, 3) => 45
      | (3, 6, 5) => 44
      | (3, 6, 6) => 51
      | (4, 0, 3) => 9
      | (4, 0, 5) => 0
      | (4, 0, 6) => 55
      | (5, 0, 0) => 0
      | _ => 500
      end,
      (bindGen 
      arbitrary 
      (fun p1 => 
        (returnGen (Bool p1))))); 
      (* Abs *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 87
      | (1, 3, 5) => 64
      | (1, 3, 6) => 64
      | (1, 5, 3) => 88
      | (1, 5, 5) => 78
      | (1, 5, 6) => 75
      | (1, 6, 3) => 86
      | (1, 6, 5) => 47
      | (1, 6, 6) => 73
      | (2, 3, 3) => 28
      | (2, 3, 5) => 12
      | (2, 3, 6) => 3
      | (2, 5, 3) => 73
      | (2, 5, 5) => 71
      | (2, 5, 6) => 60
      | (2, 6, 3) => 4
      | (2, 6, 5) => 18
      | (2, 6, 6) => 28
      | (3, 3, 3) => 70
      | (3, 3, 5) => 50
      | (3, 3, 6) => 60
      | (3, 5, 3) => 3
      | (3, 5, 5) => 43
      | (3, 5, 6) => 3
      | (3, 6, 3) => 60
      | (3, 6, 5) => 60
      | (3, 6, 6) => 55
      | (4, 0, 3) => 88
      | (4, 0, 5) => 0
      | (4, 0, 6) => 82
      | (5, 0, 0) => 96
      | _ => 500
      end,
      (bindGen (genTyp 2 stack2 2) 
      (fun p1 => 
        (bindGen (genExpr size1 stack2 5) 
        (fun p2 => 
          (returnGen (Abs p1 p2))))))); 
      (* App *) (match (size, stack1, stack2) with
      | (1, 3, 3) => 83
      | (1, 3, 5) => 47
      | (1, 3, 6) => 71
      | (1, 5, 3) => 45
      | (1, 5, 5) => 25
      | (1, 5, 6) => 34
      | (1, 6, 3) => 70
      | (1, 6, 5) => 83
      | (1, 6, 6) => 48
      | (2, 3, 3) => 93
      | (2, 3, 5) => 84
      | (2, 3, 6) => 93
      | (2, 5, 3) => 58
      | (2, 5, 5) => 54
      | (2, 5, 6) => 42
      | (2, 6, 3) => 94
      | (2, 6, 5) => 88
      | (2, 6, 6) => 92
      | (3, 3, 3) => 48
      | (3, 3, 5) => 59
      | (3, 3, 6) => 39
      | (3, 5, 3) => 95
      | (3, 5, 5) => 82
      | (3, 5, 6) => 95
      | (3, 6, 3) => 49
      | (3, 6, 5) => 53
      | (3, 6, 6) => 49
      | (4, 0, 3) => 30
      | (4, 0, 5) => 96
      | (4, 0, 6) => 17
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
          
